#ifndef LLM_LIFECYCLE_NODE_HPP
#define LLM_LIFECYCLE_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp/utilities.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <memory>
#include <string>
#include <vector>
#include <regex>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <mutex>

#include "llama.h"
#include "buddy_interfaces/msg/person_response.hpp"
#include "buddy_interfaces/msg/llm_status.hpp"
#include "buddy_interfaces/action/process_response.hpp"

namespace buddy_chat {

class TextProcessor {
public:
    static std::vector<std::string> clean_text(const std::string& text);
};

class LLMLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
    using GoalHandleProcessResponse = rclcpp_action::ServerGoalHandle<buddy_interfaces::action::ProcessResponse>;

    explicit LLMLifecycleNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~LLMLifecycleNode();

private:
    // ROS interfaces
    rclcpp::Subscription<buddy_interfaces::msg::PersonResponse>::SharedPtr subscription_;
    rclcpp::Publisher<buddy_interfaces::msg::LLMStatus>::SharedPtr llm_status_publisher_;
    rclcpp_action::Server<buddy_interfaces::action::ProcessResponse>::SharedPtr action_server_;
    std::mutex llama_mutex_;
    
    // LLM model
    std::string model_path_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    
    struct ChatMessage {
        std::string role;
        std::string content;
    };
    std::vector<ChatMessage> conversation_history_;

    // Lifecycle methods
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State& state);

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State& state);

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State& state);

    void process_input(const buddy_interfaces::msg::PersonResponse::SharedPtr msg);

    // Action Server methods
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const buddy_interfaces::action::ProcessResponse::Goal> goal);

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleProcessResponse> goal_handle);

    void handle_accepted(const std::shared_ptr<GoalHandleProcessResponse> goal_handle);

    void execute_response_generation(const std::shared_ptr<GoalHandleProcessResponse> goal_handle);
};

} // namespace buddy_chat

#endif // LLM_LIFECYCLE_NODE_HPP