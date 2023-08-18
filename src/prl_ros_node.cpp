#include<iostream>

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

	std::string open_ts_name = node_handle.param("/prl/open_ts_name", std::string());
	std::string save_ts_name = node_handle.param("/prl/save_ts_name", std::string());

	/////////////////   Transition System   /////////////////
	
	GF::DiscreteModel::ManipulatorModelProperties ts_props;
	node_handle.getParam("/environment/location_names", ts_props.locations);
	node_handle.getParam("/objects/object_ids", ts_props.objects);

	// TODO: remove assumption that init ee location is stow	
	ts_props.init_ee_location = GF::DiscreteModel::ManipulatorModelProperties::s_stow;
	std::vector<std::string> obj_locations = getObjectLocations(node_handle, ts_props.objects);
	for (uint32_t i = 0; i < ts_props.objects.size(); ++i) {
		// Number of objects must match the number of locations returned from getObjectLocations
		ts_props.init_obj_locations[ts_props.objects[i]] = obj_locations[i];
	}

	std::shared_ptr<GF::DiscreteModel::TransitionSystem> ts;

	// If opening an existing ts, check if the model properties are equivalent
	GFROS::ManipulatorTSCacheHandler cache_handler(node_handle);
	if (open_ts_name.empty() || !cache_handler.get(open_ts_name, ts_props, ts)) {
		// If TS was not generated, that means a cached file was not found, so generate
		ROS_INFO_STREAM("Generating transition system...");
		ts = GF::DiscreteModel::Manipulator::generate(ts_props);
		ROS_INFO_STREAM("Done!");
	}

	if (!save_ts_name.empty()) {
		LOG("MAKING TS");
		cache_handler.make(save_ts_name, ts_props, *ts);
	}

	if (show_propositions_only) {
		ts->listPropositions();
		return 0;
	}

	/////////////////   DFAs   /////////////////

 	std::vector<GF::FormalMethods::DFAptr> dfas(1);
	dfas[0].reset(new GF::FormalMethods::DFA());
	dfas[0]->generateFromFormula(formula);

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
		

	std::shared_ptr<PRL::DataCollector<N>> data_collector = std::make_shared<DataCollector<N>>(product, p_ev);

	std::shared_ptr<BehaviorHandlerType> behavior_handler;
	behavior_handler = std::make_shared<BehaviorHandlerType>(product, 1, confidence, default_mean_converted);

	PRL::Learner<N> prl(behavior_handler, data_collector, n_efe_samples, true);

	// Initialize the agent's state
	GF::DiscreteModel::State init_state = GF::DiscreteModel::Manipulator::makeInitState(ts_props, ts);
	prl.initialize(init_state);

	// Make the action caller
	GFROS::ActionCaller action_caller(node_handle, product, risk_object_id, true);

	// Run the PRL
	LOG("Running...");
	prl.run(p_ev, action_caller, max_planning_instances, selector);

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

