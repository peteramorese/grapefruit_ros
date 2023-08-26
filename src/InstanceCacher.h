#pragma once

#include <ros/ros.h>

#include <filesystem>

#include "Grapefruit.h"

#include "src/pareto_reinforcement_learning/BehaviorHandler.h"

namespace GFROS {

class InstanceCacheHandler {
    public:
        using EdgeInheritor = GF::DiscreteModel::ModelEdgeInheritor<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA>;
        using SymbolicGraph = GF::DiscreteModel::SymbolicProductAutomaton<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA, EdgeInheritor>;
        using BehaviorHandlerType = PRL::BehaviorHandler<SymbolicGraph, 2>;
    public:
        InstanceCacheHandler(const ros::NodeHandle& nh) {
            nh.getParam("cache_dir", m_fp);
            m_fp.push_back('/');
        }

        void make(const std::string& name, const PRL::BehaviorHandler<SymbolicGraph, 2>& behavior_handler, uint32_t instance, const GF::DiscreteModel::State& instance_final_state) {
            GF::Serializer szr(m_fp + name + "_instance_" + std::to_string(instance) + ".yaml");
            YAML::Emitter& out = szr.get();

            out << YAML::Key << "Instance State" << YAML::Value << YAML::BeginMap;
            instance_final_state.serialize(szr);
            out << YAML::EndMap;

            out << YAML::Key << "Behavior Handler" << YAML::Value << YAML::BeginMap;
            behavior_handler.serialize(szr);
            out << YAML::EndMap;

            szr.done();
        }

        bool get(const std::string& name, PRL::BehaviorHandler<SymbolicGraph, 2>& behavior_handler, uint32_t instance, GF::DiscreteModel::State* instance_final_state = nullptr) {
            if (std::filesystem::exists(m_fp + name + "_instance_" + std::to_string(instance) + ".yaml")) {
                GF::Deserializer dszr(m_fp + name + "_instance_" + std::to_string(instance) + ".yaml");

                if (instance_final_state) 
                    instance_final_state->deserialize(dszr.get()["Instance State"]);

                behavior_handler.deserialize(dszr.get()["Behavior Handler"]);
                return true;
            } else {
                ROS_ERROR_STREAM("Did not find saved starting instance '" << instance << "' under name '" << name << "'");
                return false;
            }
        }

    private:
        std::string m_fp = std::string();

};



}