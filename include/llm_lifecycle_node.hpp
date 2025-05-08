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
#include "coco_interfaces/action/process_response.hpp"

namespace coco_chat {

class TextProcessor {
public:
    static std::vector<std::string> clean_text(const std::string& text);
};

class LLMLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
    using GoalHandleProcessResponse = rclcpp_action::ServerGoalHandle<coco_interfaces::action::ProcessResponse>;

    explicit LLMLifecycleNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~LLMLifecycleNode();

private:
    rclcpp_action::Server<coco_interfaces::action::ProcessResponse>::SharedPtr action_server_;
    std::mutex llama_mutex_;
    
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
    std::vector<ChatMessage> conversation_history_for_summary_;

    std::string system_prompt_base_ = "Eres Coco, un asistente virtual amigable para niños que se encuentran en hospitales "
                                    "y tienen entre 7 a 12 años.";

    const float CONTEXT_USAGE_THRESHOLD_ = 0.9f; 

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State& state);

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State& state);

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State& state);

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const coco_interfaces::action::ProcessResponse::Goal> goal);

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleProcessResponse> goal_handle);

    void handle_accepted(const std::shared_ptr<GoalHandleProcessResponse> goal_handle);

    void execute_response_generation(const std::shared_ptr<GoalHandleProcessResponse> goal_handle);

    bool manage_context(int tokens_to_add);

    std::string generate_conversation_summary();
};

} // namespace coco_chat

#endif // LLM_LIFECYCLE_NODE_HPP