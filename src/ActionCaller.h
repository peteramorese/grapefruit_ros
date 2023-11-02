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
        ActionCaller(ros::NodeHandle& nh, const std::shared_ptr<SymbolicGraph>& product, const std::string& risk_obj_id, bool use_linear_actions = true) 
            : m_product(product) 
            , m_risk_obj_id(risk_obj_id)
        {
            update_env_client = nh.serviceClient<taskit::UpdateEnv>("/manipulator_node/action_primitive/update_environment");

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
            const GF::DiscreteModel::State& src_state = m_product->getModel().getGenericNodeContainer()[src_model_node];
            const GF::DiscreteModel::State& dst_state = m_product->getModel().getGenericNodeContainer()[dst_model_node];

            GF::Containers::FixedArray<N, float> cost_sample;

            // Before execution the action, update the environment:
            updateEnvironment();


            if (action.starts_with("grasp")) {
                ROS_INFO("Calling action: GRASP");
                if (!grasp(cost_sample)) {
                    ROS_ASSERT_MSG(false, "Grasp failed, killing...");
                }
            } else if (action.starts_with("release")) {
                ROS_INFO("Calling action: RELEASE");
                if (!release(cost_sample)) {
                    ROS_ASSERT_MSG(false, "Release failed, killing...");
                }
            } else if (action.starts_with("transit")) {
                const std::string& ee_loc = dst_state["ee_loc"];
                ROS_INFO_STREAM("Calling action: TRANSIT (effector destination location: " << ee_loc << ")");
                if (!transit(cost_sample, ee_loc)) {
                    ROS_ASSERT_MSG(false, "Transit failed, killing...");
                }
            } else if (action.starts_with("transport")) {
                // Check if the risky object is being held, and risk needs to be calculated for the action
                bool isHoldingRiskObject = src_state[m_risk_obj_id] == "ee";
                if (isHoldingRiskObject)
                    ROS_INFO_STREAM("Transporting risky object!");

                const std::string& ee_loc = dst_state["ee_loc"];
                ROS_INFO_STREAM("Calling action: TRANSPORT (effector destination location: " << ee_loc << ")");
                if (!transport(cost_sample, ee_loc, isHoldingRiskObject)) {
                    ROS_ASSERT_MSG(false, "Transport failed, killing...");
                }
            } else {
                ROS_ASSERT_MSG(false, "Unrecognized action");
            }
            ROS_INFO_STREAM("Action cost 0 (total execution time)    : " << cost_sample[0]);
            ROS_INFO_STREAM("Action cost 1 (total risk)              : " << cost_sample[1]);
            return cost_sample;
        }

        /// @brief Calculate the risk of a trajectory, assuming the eef is holding an object. The risk is the height of the 
        /// end effector above the ground integrated over time
        /// @param eef_trajectory Sequence of eef poses along trajectory
        /// @param waypoint_durations Time duration of each pose
        /// @return Total risk
        static double getHeightAboveGroundRisk(const std::vector<geometry_msgs::Pose>& eef_trajectory, const std::vector<double>& waypoint_durations, double execution_time) {
            auto riskFunction = [](double time, double height_m) -> double {
                return 0.01 * time * std::pow(10.0 * height_m, 2);
                //return time * height_m;
            };

            double risk = 0.0;
            double movement_time = 0.0;
            for (std::size_t i = 0; i < eef_trajectory.size(); ++i) {
                risk += riskFunction(waypoint_durations[i], eef_trajectory[i].position.z);
                movement_time += waypoint_durations[i];
            }

            // Add the risk for holding the object during planning
            double planning_time = execution_time - movement_time; // Planning time is the total time (execution) minus the time spent moving

            ROS_INFO_STREAM("Execution time: " << execution_time << " Planning time: " << planning_time);
            ROS_INFO_STREAM("Risk from movement: " << risk);
            ROS_INFO_STREAM("Risk from holding: " << riskFunction(planning_time, eef_trajectory[0].position.z));

            // Holding risk before trajectory
            risk += riskFunction(0.5 * planning_time, eef_trajectory.front().position.z);

            // Holding risk after trajectory
            risk += riskFunction(0.5 * planning_time, eef_trajectory.back().position.z);

            return risk;
        }

    private:
        
        bool updateEnvironment() {
            taskit::UpdateEnv srv;
            srv.request.include_static = false; // Empty object ID will pick up whatver object is in the EEF location
            if (update_env_client.call(srv)) {
                if (!srv.response.found_all) {
                    ROS_ERROR("Did not find all objects when updating environment");
                    return false;
                }
                return true;
            } else {
                ROS_ERROR("Failed to call 'update_environment' service");
                return false;
            }
        }

        bool grasp(GF::Containers::FixedArray<N, float>& cost_sample) {
            taskit::Grasp srv;
            srv.request.obj_id = std::string(); // Empty object ID will pick up whatver object is in the EEF location
            if (grasp_client.call(srv)) {
                if (srv.response.mv_props.execution_success) {
                    cost_sample[0] = srv.response.mv_props.execution_time;
                    cost_sample[1] = 0.0f;
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
                    cost_sample[1] = 0.0f;
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
                    cost_sample[1] = 0.0f;
                    return true;
                } else {
                    return false;
                }
            } else {
                ROS_ERROR("Failed to call 'transit' service");
                return false;
            }
        }

        bool transport(GF::Containers::FixedArray<N, float>& cost_sample, const std::string& dst_location, bool get_risk) {
            taskit::Transit srv;
            srv.request.destination_location = dst_location;
            if (transport_client.call(srv)) {
                if (srv.response.mv_props.execution_success) {
                    cost_sample[0] = srv.response.mv_props.execution_time;
                    cost_sample[1] = (get_risk) ? getHeightAboveGroundRisk(srv.response.mv_props.eef_trajectory, srv.response.mv_props.waypoint_durations, srv.response.mv_props.execution_time) : 0.0f;
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
        std::string m_risk_obj_id;

        ros::ServiceClient update_env_client;
        ros::ServiceClient grasp_client;
        ros::ServiceClient release_client;
        ros::ServiceClient transit_client;
        ros::ServiceClient transport_client;
};

}
