#include <memory>
#include <thread>
#include <mutex>
#include <string>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "std_msgs/msg/bool.hpp"
#include "lifecycle_msgs/msg/transition.hpp"
#include "lifecycle_msgs/srv/change_state.hpp"

class LifecycleNodesManager : public rclcpp::Node {
public:
    LifecycleNodesManager() : Node("lifecycle_nodes_manager") {
        // Create subscription
        stt_status_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/stt_terminado", 10, 
            std::bind(&LifecycleNodesManager::stt_status_callback, this, std::placeholders::_1));
        
        // Create clients
        stt_state_client_ = this->create_client<lifecycle_msgs::srv::ChangeState>(
            "/stt_lifecycle_node/change_state");
        llm_state_client_ = this->create_client<lifecycle_msgs::srv::ChangeState>(
            "/llm_lifecycle_node/change_state");
        tts_state_client_ = this->create_client<lifecycle_msgs::srv::ChangeState>(
            "/tts_lifecycle_node/change_state");
        
        stt_terminated_ = false;
        
        // Configure initial nodes
        _configure_initial_nodes();
    }

private:
    // Subscribers and clients
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stt_status_sub_;
    rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedPtr stt_state_client_;
    rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedPtr llm_state_client_;
    rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedPtr tts_state_client_;
    
    // Variables
    bool stt_terminated_;
    std::mutex state_lock_;

    void _configure_initial_nodes() {
        // Configure STT and TTS nodes on startup
        std::thread t(&LifecycleNodesManager::_initial_configuration, this);
        t.detach();
    }

    void _initial_configuration() {
        change_node_state("/stt_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
        change_node_state("/stt_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
        change_node_state("/llm_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
        change_node_state("/tts_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
    }

    void stt_status_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_lock_);
        stt_terminated_ = msg->data;
        manage_node_lifecycle();
    }

    void manage_node_lifecycle() {
        std::thread t(&LifecycleNodesManager::_manage_lifecycle_thread, this);
        t.detach();
    }

    void _manage_lifecycle_thread() {
        std::lock_guard<std::mutex> lock(state_lock_);
        if (stt_terminated_) {
            change_node_state("/llm_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
            change_node_state("/tts_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
            change_node_state("/stt_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
        } else {
            change_node_state("/stt_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
            change_node_state("/tts_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
            change_node_state("/llm_lifecycle_node", lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
        }
    }

    void change_node_state(const std::string& node_name, uint8_t transition_id) {
        // Asynchronous node state change
        auto request = std::make_shared<lifecycle_msgs::srv::ChangeState::Request>();
        request->transition.id = transition_id;
        
        rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedPtr client;
        
        if (node_name.find("stt") != std::string::npos) {
            client = stt_state_client_;
        } else if (node_name.find("llm") != std::string::npos) {
            client = llm_state_client_;
        } else if (node_name.find("tts") != std::string::npos) {
            client = tts_state_client_;
        }
        
        std::thread t([this, client, request, node_name]() {
            if (!client->wait_for_service(std::chrono::seconds(5))) {
                RCLCPP_ERROR(this->get_logger(), "Service not available for %s", node_name.c_str());
                return;
            }
            
            auto future = client->async_send_request(request);
            
            // No usamos spin_until_future_complete para evitar problemas con el ejecutor
            // En su lugar, esperamos con un timeout
            auto status = future.wait_for(std::chrono::seconds(5));
            if (status != std::future_status::ready) {
                RCLCPP_ERROR(this->get_logger(), "Failed to call service for %s (timeout)", node_name.c_str());
            }
        });
        t.detach();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LifecycleNodesManager>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}