#include<iostream>
#include<algorithm>

// ROS
#include "ros/ros.h"

// Grapefruit
#include "grapefruit/Grapefruit.h"

// PRL
#include "src/pareto_reinforcement_learning/BehaviorHandler.h"
#include "src/pareto_reinforcement_learning/Learner.h"
#include "src/pareto_reinforcement_learning/TrueBehavior.h"
#include "src/pareto_reinforcement_learning/Misc.h"

#include "taskit/GetObjectLocations.h"

// GFROS
#include "ActionCaller.h"
#include "TSCacher.h"
#include "InstanceCacher.h"

static const std::string node_name = "prl_ros_node";
constexpr uint64_t N = 2;

using namespace PRL;

// Currently assumes that the end effector location is 'stow'
std::vector<std::string> getObjectLocations(ros::NodeHandle& nh, const std::vector<std::string>& object_ids) {
	ros::ServiceClient client = nh.serviceClient<taskit::GetObjectLocations>("/manipulator_node/action_primitive/get_object_locations");
	taskit::GetObjectLocations srv;
	srv.request.object_ids = object_ids;
	if (client.call(srv)) {
		if (srv.response.found_all) {
			return srv.response.object_locations;
		} else {
			ROS_ERROR("Not all objects were found from 'get_object_locations' service");
			return {};
		}
	} else {
		ROS_ERROR("Failed to call 'get_object_locations' service");
		return {};
	}
}

GF::Stats::Distributions::FixedMultivariateNormal<N> makePreferenceDist(const std::vector<float>& mean, const std::vector<float>& minimal_covariance) {
	ASSERT(mean.size() == N, "Mean (" << mean.size() <<") dimension does not match compile-time dimension (" << N << ")");
	ASSERT(minimal_covariance.size() == GF::Stats::Distributions::FixedMultivariateNormal<N>::uniqueCovarianceElements(), 
		"Mean (" << minimal_covariance.size() <<") dimension does not match compile-time minimal covariance dimension (" << GF::Stats::Distributions::FixedMultivariateNormal<N>::uniqueCovarianceElements() << ")");

	// Convert to Eigen
	Eigen::Matrix<float, N, 1> mean_converted;
	Eigen::Matrix<float, GF::Stats::Distributions::FixedMultivariateNormal<N>::uniqueCovarianceElements(), 1> minimal_cov_converted;
	for (uint32_t i = 0; i < N; ++i)
		mean_converted(i) = mean[i];

	for (uint32_t i = 0; i < GF::Stats::Distributions::FixedMultivariateNormal<N>::uniqueCovarianceElements(); ++i)
		minimal_cov_converted(i) = minimal_covariance[i];
	
	GF::Stats::Distributions::FixedMultivariateNormal<N> dist;
	dist.mu = mean_converted;
	dist.setSigmaFromUniqueElementVector(minimal_cov_converted);
	//ASSERT(GF::isCovariancePositiveSemiDef(dist.Sigma), "Preference Covariance: \n" << dist.Sigma <<"\nis not positive semi-definite");
	return dist;
}

std::pair<bool, Eigen::Matrix<float, N, 1>> deserializeDefaultMean(const std::string& config_filepath) {
    YAML::Node data;
    try {
        data = YAML::LoadFile(config_filepath);

        if (!data["Default Transition Estimate Mean"]) 
            return std::make_pair(false, Eigen::Matrix<float, N, 1>());

        YAML::Node pref_node = data["PRL Preference"]; 
        std::vector<float> mean = data["Default Transition Estimate Mean"].as<std::vector<float>>();

        // Convert to Eigen
        Eigen::Matrix<float, N, 1> mean_converted;
        Eigen::Matrix<float, GF::Stats::Distributions::FixedMultivariateNormal<N>::uniqueCovarianceElements(), 1> minimal_cov_converted;
        for (uint32_t i = 0; i < N; ++i)
            mean_converted(i) = mean[i];

        return std::make_pair(true, mean_converted);
    } catch (YAML::ParserException e) {
        ERROR("Failed to load file" << config_filepath << " ("<< e.what() <<")");
        return std::make_pair(false, Eigen::Matrix<float, N, 1>());
    }
}

PRL::Selector getSelector(const std::string& label) {
	if (label == "aif" || label == "Aif") {
		return Selector::Aif;
	} else if (label == "uniform" || label == "Uniform") {
		return Selector::Uniform;
	} else if (label == "topsis" || label == "TOPSIS") {
		return Selector::Topsis;
	} else if (label == "weights" || label == "Weights") {
		return Selector::Weights;
	} 
	ASSERT(false, "Unrecognized selector '" << label <<"'");
}

int main(int argc, char** argv) {
	
    ros::init(argc, argv, node_name);

    ros::NodeHandle node_handle("~");

	GFROS::waitForManipulatorNodeServices();

	// Get the prl params from the rosparam server

	bool show_propositions_only = node_handle.param("/prl/show_propositions_only", false);
	
	bool exclude_plans = node_handle.param("/prl/exclude_plans", false);

	std::string formula;
	node_handle.getParam("/prl/formula", formula);

	int max_planning_instances;
	node_handle.getParam("/prl/instances", max_planning_instances);

	int n_efe_samples;
	node_handle.getParam("/prl/n_efe_samples", n_efe_samples);
	
	float confidence;
	node_handle.getParam("/prl/confidence", confidence);

	std::vector<float> pref_mean;
	node_handle.getParam("/prl/pref_mean", pref_mean);

	std::vector<float> pref_minimal_covariance;
	node_handle.getParam("/prl/pref_minimal_covariance", pref_minimal_covariance);

	std::vector<float> default_mean;
	node_handle.getParam("/prl/default_mean", default_mean);

	std::string selector_label = node_handle.param<std::string>("/prl/selector", "aif");
	Selector selector = getSelector(selector_label);

	std::string risk_object_id;
	node_handle.getParam("/prl/risk_object_id", risk_object_id);

	std::string open_scenario_name = node_handle.param("/prl/open_scenario_name", std::string());
	std::string save_scenario_name = node_handle.param("/prl/save_scenario_name", std::string());

	std::vector<int> save_instances = node_handle.param("/prl/save_instances", std::vector<int>());
	
	int start_instance = node_handle.param("/prl/start_instance", 0);
	bool use_saved_start_instance_init_state = node_handle.param("/prl/use_saved_start_instance_init_state", false);

	/////////////////   Transition System   /////////////////
	
	GF::DiscreteModel::ManipulatorModelProperties ts_props;
	node_handle.getParam("/environment/location_names", ts_props.locations);
	node_handle.getParam("/objects/object_ids", ts_props.objects);

	// Make init ee location the first location (it does not matter)
	ts_props.init_ee_location = ts_props.locations.front();
	ts_props.include_stow = false;
	std::vector<std::string> obj_locations = getObjectLocations(node_handle, ts_props.objects);
	for (uint32_t i = 0; i < ts_props.objects.size(); ++i) {
		// Number of objects must match the number of locations returned from getObjectLocations
		ts_props.init_obj_locations[ts_props.objects[i]] = obj_locations[i];
	}

	std::shared_ptr<GF::DiscreteModel::TransitionSystem> ts;

	// If opening an existing ts, check if the model properties are equivalent
	GFROS::ManipulatorTSCacheHandler ts_cache_handler(node_handle);
	if (open_scenario_name.empty() || !ts_cache_handler.get(open_scenario_name, ts_props, ts)) {
		// If TS was not generated, that means a cached file was not found, so generate
		ROS_INFO_STREAM("Generating transition system...");
		ts = GF::DiscreteModel::Manipulator::generate(ts_props);
		ROS_INFO_STREAM("Done!");
	}

	if (!save_scenario_name.empty()) {
		ts_cache_handler.make(save_scenario_name, ts_props, *ts);
	}

	if (show_propositions_only) {
		ts->listPropositions();
		return 0;
	}

	/////////////////   DFAs   /////////////////

 	std::vector<GF::FormalMethods::DFAptr> dfas(1);
	dfas[0].reset(new GF::FormalMethods::DFA());
	dfas[0]->generateFromFormula(formula);
	dfas[0]->print();

	ts->addAlphabet(dfas[0]->getAlphabet());

	/////////////////   Planner   /////////////////

	using EdgeInheritor = GF::DiscreteModel::ModelEdgeInheritor<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA>;
	using SymbolicGraph = GF::DiscreteModel::SymbolicProductAutomaton<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA, EdgeInheritor>;
	using BehaviorHandlerType = PRL::BehaviorHandler<SymbolicGraph, N>;
	using PreferenceDistributionType = GF::Stats::Distributions::FixedMultivariateNormal<N>;

 	std::shared_ptr<SymbolicGraph> product = std::make_shared<SymbolicGraph>(ts, dfas);

	// Make the preference behavior distribution
	PreferenceDistributionType p_ev = makePreferenceDist(pref_mean, pref_minimal_covariance);

	// Get the default transition mean if the file contains it
	Eigen::Matrix<float, N, 1> default_mean_converted;
	for (uint32_t i = 0; i < N; ++i) {
		default_mean_converted(i) = default_mean[i];
	}
		
	// Make the init state
	GF::DiscreteModel::State init_state = GF::DiscreteModel::Manipulator::makeInitState(ts_props, ts);
	LOG("Init state:");
	init_state.print();

	std::shared_ptr<PRL::DataCollector<N>> data_collector = std::make_shared<DataCollector<N>>(product, p_ev);

	std::shared_ptr<BehaviorHandlerType> behavior_handler;

	// If start instance is 0, start a fresh experiement
	GFROS::InstanceCacheHandler bh_cache_handler(node_handle);
	if (start_instance == 0) {
		behavior_handler = std::make_shared<BehaviorHandlerType>(product, 1, confidence, default_mean_converted);
	} else {
		behavior_handler = std::make_shared<BehaviorHandlerType>(product, 1, confidence);
		GF::DiscreteModel::State* init_state_ptr = use_saved_start_instance_init_state ? &init_state : nullptr;
		if (!bh_cache_handler.get(open_scenario_name, *behavior_handler, start_instance, init_state_ptr)) {
			// Exit, start instance was not found
			return 1;
		}
	}
	PRL::Learner<N> prl(behavior_handler, data_collector, n_efe_samples, true);

	// Initialize the agent's state
	prl.initialize(init_state);

	// Make the action caller
	GFROS::ActionCaller action_caller(node_handle, product, risk_object_id, true);

	// Run the PRL
	if (save_instances.empty()) {
		LOG("Running...");
		prl.run(p_ev, action_caller, max_planning_instances, selector);
	} else if (save_instances.size() == 1) {
		prl.run(p_ev, action_caller, max_planning_instances, selector);
		bh_cache_handler.make(save_scenario_name, *behavior_handler, save_instances[0], prl.getCurrentState());
	} else {
		std::sort(save_instances.begin(), save_instances.end());
		ROS_ASSERT_MSG(save_instances.front() > start_instance, "Must specify save instances that are greater than the 'start_instance'");
		ROS_ASSERT_MSG(save_instances.back() < max_planning_instances, "Largest save instance is greater than max planning instances");
		
		if (save_instances.back() < (max_planning_instances - 1))
			save_instances.push_back(max_planning_instances);

		for (auto save_inst_it = save_instances.begin(); (save_inst_it + 1) != save_instances.end(); ++save_inst_it) {
			uint32_t instance_i = *save_inst_it;
			uint32_t instance_f = *(save_inst_it + 1);
			uint32_t n_instances = instance_f - instance_i + 1;
			prl.run(p_ev, action_caller, n_instances, selector);
			// Save at the final instance
			bh_cache_handler.make(save_scenario_name, *behavior_handler, instance_f, prl.getCurrentState());
		}
		
	}

	LOG("Finished!");
	std::string total_cost_str{};
	for (uint32_t i = 0; i < N; ++i) 
		total_cost_str += std::to_string(data_collector->cumulativeCost()[i]) + ((i < N) ? ", " : "");
	PRINT_NAMED("Total Cost...............................", total_cost_str);
	PRINT_NAMED("Steps....................................", data_collector->steps());
	PRINT_NAMED("Instances................................", data_collector->numInstances());
	std::string avg_cost_str{};
	auto avg_cost_per_instance = data_collector->avgCostPerInstance();
	for (uint32_t i = 0; i < N; ++i) 
		avg_cost_str += std::to_string(avg_cost_per_instance[i]) + ((i < N) ? ", " : "");
	PRINT_NAMED("Average cost per per instance............", avg_cost_str);
	

	return 0;
}

