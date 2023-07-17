#include<iostream>

#include "ros/ros.h"

#include "grapefruit/Grapefruit.h"

#include "src/pareto_reinforcement_learning/BehaviorHandler.h"
#include "src/pareto_reinforcement_learning/Learner.h"
#include "src/pareto_reinforcement_learning/TrueBehavior.h"
#include "src/pareto_reinforcement_learning/Misc.h"

static const std::string node_name = "prl_ros_node";

using namespace PRL;

int main(int argc, char** argv) {
	
    ros::init(argc, argv, node_name);

    ros::NodeHandle node_handle("~");

	Selector selector = getSelector(selector_label.get());
	
	/////////////////   Transition System   /////////////////
	
	GF::DiscreteModel::GridWorldAgentProperties ts_props;
	if (!config_filepath.has()) {
		ts_props.n_x = 10;
		ts_props.n_y = 10;
		ts_props.init_coordinate_x = 0;
		ts_props.init_coordinate_y = 0;
	} else {
		ts_props = GF::DiscreteModel::GridWorldAgent::deserializeConfig(config_filepath.get());
	}

	std::shared_ptr<GF::DiscreteModel::TransitionSystem> ts = GF::DiscreteModel::GridWorldAgent::generate(ts_props);

	ts->print();

	/////////////////   DFAs   /////////////////

	GF::Deserializer dszr(formula_filepath.get());
	auto dfas = GF::FormalMethods::createDFAsFromFile(dszr);

	GF::FormalMethods::Alphabet combined_alphbet;
	for (const auto& dfa : dfas) {
		combined_alphbet = combined_alphbet + dfa->getAlphabet();
	}


	ts->addAlphabet(combined_alphbet);

	/////////////////   Planner   /////////////////

	constexpr uint64_t N = 2;
	using EdgeInheritor = GF::DiscreteModel::ModelEdgeInheritor<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA>;
	using SymbolicGraph = GF::DiscreteModel::SymbolicProductAutomaton<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA, EdgeInheritor>;
	using BehaviorHandlerType = BehaviorHandler<SymbolicGraph, N>;
	using PreferenceDistributionType = GF::Stats::Distributions::FixedMultivariateNormal<N>;

 	std::shared_ptr<SymbolicGraph> product = std::make_shared<SymbolicGraph>(ts, dfas);

	/////////////////   True Behavior   /////////////////

	std::shared_ptr<GridWorldTrueBehavior<N>> true_behavior = std::make_shared<GridWorldTrueBehavior<N>>(product, config_filepath.get());

	// Make the preference behavior distribution
	PreferenceDistributionType p_ev = deserializePreferenceDist<N>(config_filepath.get());

	// Get the default transition mean if the file contains it
	std::pair<bool, Eigen::Matrix<float, N, 1>> default_mean = deserializeDefaultMean<N>(config_filepath.get());

	for (uint32_t trial = 0; trial < n_trials.get(); ++trial) {
		
		std::shared_ptr<Regret<SymbolicGraph, N>> regret_handler;
		if (calc_regret)
			regret_handler = std::make_shared<Regret<SymbolicGraph, N>>(product, true_behavior);

		std::shared_ptr<DataCollector<N>> data_collector = std::make_shared<DataCollector<N>>(product, p_ev, regret_handler);

		std::shared_ptr<BehaviorHandlerType> behavior_handler;
		if (default_mean.first)
			behavior_handler = std::make_shared<BehaviorHandlerType>(product, 1, confidence.get(), default_mean.second);
		else 
			behavior_handler = std::make_shared<BehaviorHandlerType>(product, 1, confidence.get());

		Learner<N> prl(behavior_handler, data_collector, n_efe_samples.get(), verbose);

		// Initialize the agent's state
		GF::DiscreteModel::State init_state = GF::DiscreteModel::GridWorldAgent::makeInitState(ts_props, ts);
		prl.initialize(init_state);

		auto samplerFunction = [&](GF::WideNode src_node, GF::WideNode dst_node, const GF::DiscreteModel::Action& action) {
			return true_behavior->sample(src_node, dst_node, action);
		};

		// Run the PRL
		prl.run(p_ev, samplerFunction, max_planning_instances.get(), selector);

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
	}
	

	return 0;
}

