#pragma once

#include "grapefruit/Grapefruit.h"

#include "taskit/StowSrv.h"
#include "taskit/GraspSrv.h"
#include "taskit/ReleaseSrv.h"
#include "taskit/TransitSrv.h"
#include "taskit/UpdateEnvSrv.h"
#include "taskit/GetObjectLocations.h"

namespace GFROS { 

class ActionCaller {
    public: 
        using EdgeInheritor = GF::DiscreteModel::ModelEdgeInheritor<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA>;
        using SymbolicGraph = GF::DiscreteModel::SymbolicProductAutomaton<GF::DiscreteModel::TransitionSystem, GF::FormalMethods::DFA, EdgeInheritor>;
        constexpr uint64_t N = 2; // Only supports two objectives: 1) execution time and 2) risk
    public:
        ActionCaller(const ros::NodeHandle& nh, const std::shared_ptr<SymbolicGraph>& product, bool use_linear_actions = true) : m_product(product) {
            grasp_client = nh.serviceClient<taskit::GraspSrv>("/manipulator_node/action_primitive/grasp");
            release_client = nh.serviceClient<taskit::GraspSrv>("/manipulator_node/action_primitive/release");
            transit_client = nh.serviceClient<taskit::GraspSrv>("/manipulator_node/action_primitive/" + (use_linear_actions ? "linear_transit_up" : "transit_up"));
            transport_client = nh.serviceClient<taskit::GraspSrv>("/manipulator_node/action_primitive/" + (use_linear_actions ? "linear_transport" : "transport"));
        }

        GF::Containers::FixedArray<N, float> operator()(GF::WideNode src_node, GF::WideNode dst_node, const GF::DiscreteModel::Action& action) {
            GF::Node src_model_node = m_product->getUnwrappedNode(src_node).ts_node;
            GF::Node dst_model_node = m_product->getUnwrappedNode(dst_node).ts_node;

            if (action == "grasp") {

            } else if (action == "release") {

            } else if (action == "transit") {

            } else if (action == "transport") {

            }

        }

    private:
        bool grasp(GF::Containers::FixedArray<N, float>& cost_sample) {
            taskit::GraspSrv srv;
            srv.request.object_id = std::string(); // Empty object ID will pick up whatver object is in the EEF location
            if (grasp_client.call(srv)) {
                if (srv.response.success) {
                    cost_sample[0] = srv.response.execution_time;
                    cost_sample[1] = srv.response.execution_time;
                    return srv.response.object_locations;
                } else {
                    return false;
                }
            } else {
                ROS_ERROR("Failed to call 'grasp' service");
                return false;
            }
        }
        bool release(GF::Containers::FixedArray<N, float>& cost_sample) {}
        bool transit(GF::Containers::FixedArray<N, float>& cost_sample) {}
        bool transport(GF::Containers::FixedArray<N, float>& cost_sample) {}

            //taskit::GetObjectLocations srv;
            //srv.request.object_ids = object_ids;
            //if (client.call(srv)) {
            //    if (srv.response.found_all) {
            //        return srv.response.object_locations;
            //    } else {
            //        ROS_ERROR("Not all objects were found from 'get_object_locations' service");
            //        return {};
            //    }
            //} else {
            //    ROS_ERROR("Failed to call 'get_object_locations' service");
            //    return {};
            //}
    private:
        std::shared_ptr<SymbolicGraph> m_product;

        ros::ServiceClient grasp_client;
        ros::ServiceClient release_client;
        ros::ServiceClient transit_client;
        ros::ServiceClient transport_client;
};

}