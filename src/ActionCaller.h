#pragma once

#include "grapefruit/Grapefruit.h"

// Srv types
#include "taskit/Stow.h"
#include "taskit/Grasp.h"
#include "taskit/Release.h"
#include "taskit/Transit.h"
#include "taskit/UpdateEnv.h"
#include "taskit/GetObjectLocations.h"

// Msg types
#include "taskit/MovementProperties.h"



namespace GFROS { 

void waitForManipulatorNodeServices() {
    // Wait until services are active (10s)
    bool services_active = ros::service::waitForService("/manipulator_node/action_primitive/grasp", 10000);
    if (!services_active)
        ROS_ASSERT_MSG(false, "Cannot find manipulator node services");
    ros::WallDuration(1.0).sleep();
}

class ActionCaller {
    public: 
        using EdgeInheritor = GF::DiscreteModel::ModelEdgeInheritor<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA>;
        using SymbolicGraph = GF::DiscreteModel::SymbolicProductAutomaton<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA, EdgeInheritor>;
        static constexpr uint64_t N = 2; // Only supports two objectives: 1) execution time and 2) risk
    public:
        ActionCaller(ros::NodeHandle& nh, const std::shared_ptr<SymbolicGraph>& product, bool use_linear_actions = true) : m_product(product) {

            grasp_client = nh.serviceClient<taskit::Grasp>("/manipulator_node/action_primitive/grasp");
            release_client = nh.serviceClient<taskit::Release>("/manipulator_node/action_primitive/release");

            std::string transit_service, transport_service;
            if (use_linear_actions) {
                transit_service = "/manipulator_node/action_primitive/linear_transit_up";
                transport_service = "/manipulator_node/action_primitive/linear_transport";
            } else {
                transit_service = "/manipulator_node/action_primitive/transit_up";
                transport_service = "/manipulator_node/action_primitive/transport";
            }
            transit_client = nh.serviceClient<taskit::Transit>(transit_service);
            transport_client = nh.serviceClient<taskit::Transit>(transport_service);
        }

        GF::Containers::FixedArray<N, float> operator()(const GF::WideNode& src_node, const GF::WideNode& dst_node, const GF::DiscreteModel::Action& action) {
            GF::Node src_model_node = m_product->getUnwrappedNode(src_node).ts_node;
            GF::Node dst_model_node = m_product->getUnwrappedNode(dst_node).ts_node;

            GF::Containers::FixedArray<N, float> cost_sample;

            if (action == "grasp") {
                ROS_INFO("Calling action: GRASP");
                if (!grasp(cost_sample)) {
                    ROS_ASSERT_MSG(false, "Grasp failed, killing...");
                }
            } else if (action == "release") {
                ROS_INFO("Calling action: RELEASE");
                if (!release(cost_sample)) {
                    ROS_ASSERT_MSG(false, "Release failed, killing...");
                }
            } else if (action == "transit") {
                const GF::DiscreteModel::State& dst_state = m_product->getModel().getGenericNodeContainer()[dst_model_node];
                const std::string& ee_loc = dst_state["ee_loc"];
                ROS_INFO_STREAM("Calling action: TRANSIT (effector destination location: " << ee_loc << ")");
                if (!transit(cost_sample, ee_loc)) {
                    ROS_ASSERT_MSG(false, "Transit failed, killing...");
                }
            } else if (action == "transport") {
                const GF::DiscreteModel::State& dst_state = m_product->getModel().getGenericNodeContainer()[dst_model_node];
                const std::string& ee_loc = dst_state["ee_loc"];
                ROS_INFO_STREAM("Calling action: TRANSPORT (effector destination location: " << ee_loc << ")");
                if (!transport(cost_sample, ee_loc)) {
                    ROS_ASSERT_MSG(false, "Transport failed, killing...");
                }
            }

            return cost_sample;
        }

    private:
        bool grasp(GF::Containers::FixedArray<N, float>& cost_sample) {
            taskit::Grasp srv;
            srv.request.obj_id = std::string(); // Empty object ID will pick up whatver object is in the EEF location
            if (grasp_client.call(srv)) {
                if (srv.response.mv_props.execution_success) {
                    cost_sample[0] = srv.response.mv_props.execution_time;
                    cost_sample[1] = srv.response.mv_props.max_velocity;
                    return true;
                } else {
                    return false;
                }
            } else {
                ROS_ERROR("Failed to call 'grasp' service");
                return false;
            }
        }

        bool release(GF::Containers::FixedArray<N, float>& cost_sample) {
            taskit::Release srv;
            srv.request.obj_id = std::string(); // Empty object ID will pick up whatver object is in the EEF location
            if (release_client.call(srv)) {
                if (srv.response.mv_props.execution_success) {
                    cost_sample[0] = srv.response.mv_props.execution_time;
                    cost_sample[1] = srv.response.mv_props.max_velocity;
                    return true;
                } else {
                    return false;
                }
            } else {
                ROS_ERROR("Failed to call 'release' service");
                return false;
            }
        }

        bool transit(GF::Containers::FixedArray<N, float>& cost_sample, const std::string& dst_location) {
            taskit::Transit srv;
            srv.request.destination_location = dst_location;
            if (transit_client.call(srv)) {
                if (srv.response.mv_props.execution_success) {
                    cost_sample[0] = srv.response.mv_props.execution_time;
                    cost_sample[1] = srv.response.mv_props.max_velocity;
                    return true;
                } else {
                    return false;
                }
            } else {
                ROS_ERROR("Failed to call 'transit' service");
                return false;
            }
        }

        bool transport(GF::Containers::FixedArray<N, float>& cost_sample, const std::string& dst_location) {
            taskit::Transit srv;
            srv.request.destination_location = dst_location;
            if (transport_client.call(srv)) {
                if (srv.response.mv_props.execution_success) {
                    cost_sample[0] = srv.response.mv_props.execution_time;
                    cost_sample[1] = srv.response.mv_props.max_velocity;
                    return true;
                } else {
                    return false;
                }
            } else {
                ROS_ERROR("Failed to call 'transport' service");
                return false;
            }
        }

    private:
        std::shared_ptr<SymbolicGraph> m_product;

        ros::ServiceClient grasp_client;
        ros::ServiceClient release_client;
        ros::ServiceClient transit_client;
        ros::ServiceClient transport_client;
};

}