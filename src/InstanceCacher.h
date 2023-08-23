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

        void make(const std::string& name, const PRL::BehaviorHandler<SymbolicGraph, 2>& behavior_handler, uint32_t instance) {
            GF::Serializer szr(m_fp + name + "_instance_" + std::to_string(instance) + ".yaml");

            behavior_handler.serialize(szr);
            szr.done();
        }

        bool get(const std::string& name, PRL::BehaviorHandler<SymbolicGraph, 2>& behavior_handler, uint32_t instance) {
            if (std::filesystem::exists(m_fp + name + "_instance_" + std::to_string(instance) + ".yaml")) {
                GF::Deserializer dszr(m_fp + name + "_instance_" + std::to_string(instance) + ".yaml");
                behavior_handler.deserialize(dszr);
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