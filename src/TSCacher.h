#pragma once

#include <ros/ros.h>

#include <filesystem>

#include "Grapefruit.h"

namespace GFROS {

class ManipulatorTSCacheHandler {
    public:
        ManipulatorTSCacheHandler(const ros::NodeHandle& nh) {
            nh.getParam("ts_cache_dir", m_fp);
            m_fp.push_back('/');
            ROS_INFO_STREAM("TS cache directory: " << m_fp);
        }

        void make(const std::string& name, const GF::DiscreteModel::TransitionSystemModelProperties& props, const GF::DiscreteModel::TransitionSystem& ts) {
            LOG("making ts file: " << m_fp + name + "_model.yaml");
            GF::Serializer ts_szr(m_fp + name + "_model.yaml");
            GF::Serializer props_szr(m_fp + name + "_props.yaml");

            ts.serialize(ts_szr);
            ts_szr.done();
            props.serialize(props_szr);
            props_szr.done();

        }

        bool get(const std::string& name, const GF::DiscreteModel::ManipulatorModelProperties& props, std::shared_ptr<GF::DiscreteModel::TransitionSystem>& ts) {
            if (std::filesystem::exists(m_fp + name + "_model.yaml") && std::filesystem::exists(m_fp + name + "_props.yaml")) {
                
                // Deserialize the properties to check if they match the input properties
                GF::DiscreteModel::ManipulatorModelProperties test_props;
                GF::Deserializer props_dszr(m_fp + name + "_props.yaml");
                test_props.deserialize(props_dszr);

                if (test_props.isEqual(props)) {
                    ts.reset(new GF::DiscreteModel::TransitionSystem);
                    GF::Deserializer ts_dszr(m_fp + name + "_model.yaml");
                    ts->deserialize(ts_dszr);
                } else {
                    ROS_WARN_STREAM("Properties for TS name '" << name << "' do not match, regenerating...");
                    return false;
                }
                return true;
            } else {
                ROS_WARN_STREAM("Did not find saved TS '" << name << "', regenerating...");
                return false;
            }
        }

    private:
        std::string m_fp = std::string();

};



}