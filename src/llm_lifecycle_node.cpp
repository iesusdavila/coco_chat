#include "llm_lifecycle_node.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

namespace buddy_chat {

std::vector<std::string> TextProcessor::clean_text(const std::string& text) {
    std::string cleaned = std::regex_replace(text, std::regex("[!¡?¿*,.:;()\\[\\]{}]"), " ");
    cleaned = std::regex_replace(cleaned, std::regex("\\s+"), " ");
    
    std::vector<std::string> sentences;
    // Usar regex compatible (sin lookbehind):
    std::regex sentence_regex("([.!?])\\s+");  // Detecta puntuación seguida de espacios
    
    std::sregex_token_iterator iter(cleaned.begin(), cleaned.end(), 
                                    sentence_regex, {-1, 0}); // -1: no match, 0: match
    std::sregex_token_iterator end;
    
    std::string sentence;
    for (; iter != end; ++iter) {
        sentence += iter->str();
        if (std::regex_search(iter->str(), sentence_regex)) {
            sentences.push_back(sentence);
            sentence.clear();
        }
    }
    if (!sentence.empty()) {
        sentences.push_back(sentence);
    }
    
    return sentences;
}

LLMLifecycleNode::LLMLifecycleNode(const rclcpp::NodeOptions& options)
    : LifecycleNode("llm_lifecycle_node", options) {
    
    // Get model path
    std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("buddy_chat");
    model_path_ = pkg_share_dir + "/models/LLM/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8/meta-llama-3.1-8b-instruct-q4_0.gguf";
    
    RCLCPP_INFO(get_logger(), "Model path: %s", model_path_.c_str());
}

LLMLifecycleNode::~LLMLifecycleNode() {
    // Free resources
    if (sampler_ != nullptr) {
        llama_sampler_free(sampler_);
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
    }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LLMLifecycleNode::on_configure(const rclcpp_lifecycle::State& state) {
    RCLCPP_INFO(get_logger(), "Configuring LLM Node");
    
    try {
        // Load dynamic backends
        ggml_backend_load_all();
        
        // Initialize the model
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 10;  // Same as n_gpu_layers in Python
        
        model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
        if (!model_) {
            RCLCPP_ERROR(get_logger(), "Error: unable to load model");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
        }
        
        vocab_ = llama_model_get_vocab(model_);
        
        // Initialize the context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;  // Same as n_ctx in Python
        ctx_params.n_batch = 2048;
        
        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            RCLCPP_ERROR(get_logger(), "Error: failed to create llama_context");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
        }
        
        // Initialize the sampler
        sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler_, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.7f)); // Same as temperature in Python
        llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        
        // Create subscriptions 
        subscription_ = this->create_subscription<buddy_interfaces::msg::PersonResponse>(
            "/response_person", 10, 
            std::bind(&LLMLifecycleNode::process_input, this, _1));
            
        // Create publishers
        llm_status_publisher_ = this->create_publisher<buddy_interfaces::msg::LLMStatus>(
            "/llm_status", 10);
            
        // Create action server
        action_server_ = rclcpp_action::create_server<buddy_interfaces::action::ProcessResponse>(
            this,
            "/response_llama",
            std::bind(&LLMLifecycleNode::handle_goal, this, _1, _2),
            std::bind(&LLMLifecycleNode::handle_cancel, this, _1),
            std::bind(&LLMLifecycleNode::handle_accepted, this, _1));
            
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to configure: %s", e.what());
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LLMLifecycleNode::on_activate(const rclcpp_lifecycle::State& state) {
    RCLCPP_INFO(get_logger(), "Activating LLM Node");
    
    // Activate the publisher
    // llm_status_publisher_->on_activate();
    
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LLMLifecycleNode::on_deactivate(const rclcpp_lifecycle::State& state) {
    RCLCPP_INFO(get_logger(), "Deactivating LLM Node");
    
    // Deactivate the publisher
    // llm_status_publisher_->on_deactivate();
    
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void LLMLifecycleNode::process_input(const buddy_interfaces::msg::PersonResponse::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Procesando mensaje de la persona: %s", msg->text.c_str());
    
    auto status_msg = std::make_shared<buddy_interfaces::msg::LLMStatus>();
    status_msg->is_processing = true;
    status_msg->current_response = msg->text;
    status_msg->timestamp = this->now();
    llm_status_publisher_->publish(*status_msg);
}

// Action Server methods
rclcpp_action::GoalResponse LLMLifecycleNode::handle_goal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const buddy_interfaces::action::ProcessResponse::Goal> goal)
{
    RCLCPP_INFO(get_logger(), "Received goal request with text: %s", goal->input_text.c_str());
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse LLMLifecycleNode::handle_cancel(
    const std::shared_ptr<GoalHandleProcessResponse> goal_handle)
{
    RCLCPP_INFO(get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
}

void LLMLifecycleNode::handle_accepted(const std::shared_ptr<GoalHandleProcessResponse> goal_handle)
{
    // Start a new thread to process the goal
    std::thread{std::bind(&LLMLifecycleNode::execute_response_generation, this, _1), goal_handle}.detach();
}

void LLMLifecycleNode::execute_response_generation(const std::shared_ptr<GoalHandleProcessResponse> goal_handle)
{
    RCLCPP_INFO(get_logger(), "Executing goal");

    std::lock_guard<std::mutex> lock(llama_mutex_); 
    
    auto feedback = std::make_shared<buddy_interfaces::action::ProcessResponse::Feedback>();
    auto result = std::make_shared<buddy_interfaces::action::ProcessResponse::Result>();

    if (!model_ || !ctx_ || !sampler_) {
        RCLCPP_INFO(get_logger(), "LLAMA resources not initialized");
        result->completed = false;
        goal_handle->abort(result);
        return;
    } 
    else {
        RCLCPP_INFO(get_logger(), "LLAMA resources initialized");
    }
    
    // System prompt
    const std::string system_prompt = 
        "Eres Leo, un asistente virtual amigable para niños que se encuentran en hospitales "
        "y tienen entre 7 a 12 años.";
    
    // Set up the conversation history
    conversation_history_.clear();
    conversation_history_.push_back({"system", system_prompt});
    conversation_history_.push_back({"user", goal_handle->get_goal()->input_text});
    
    // Format the messages using the chat template
    const char* tmpl = llama_model_chat_template(model_);
    std::vector<llama_chat_message> llama_messages;
    
    for (const auto& msg : conversation_history_) {
        llama_chat_message llm_msg;
        llm_msg.role = msg.role.c_str();
        llm_msg.content = msg.content.c_str();
        llama_messages.push_back(llm_msg);
    }
    
    std::vector<char> formatted(llama_n_ctx(ctx_));
    int len = llama_chat_apply_template(tmpl, llama_messages.data(), llama_messages.size(), true, formatted.data(), formatted.size());
    
    if (len < 0 || len > static_cast<int>(formatted.size())) {
        RCLCPP_ERROR(get_logger(), "Failed to apply the chat template");
        result->full_response = "Error: failed to format chat";
        result->completed = false;
        goal_handle->abort(result);
        return;
    }
    
    std::string prompt(formatted.begin(), formatted.begin() + len);
    RCLCPP_INFO(get_logger(), "Generando respuesta...");
    RCLCPP_INFO(get_logger(), "Mensaje de entrada: %s", goal_handle->get_goal()->input_text.c_str());
    
    // Tokenize the prompt
    const int n_prompt_tokens = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), llama_get_kv_cache_used_cells(ctx_) == 0, true) < 0) {
        RCLCPP_ERROR(get_logger(), "Failed to tokenize the prompt");
        result->full_response = "Error: failed to tokenize prompt";
        result->completed = false;
        goal_handle->abort(result);
        return;
    }
    
    // Prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    std::string buffer;
    std::string full_response;
    
    llama_token new_token_id;
    int max_tokens = 300; // Same as in Python
    int token_count = 0;
    
    while (token_count < max_tokens) {
        // Check if we have enough space in the context
        int n_ctx = llama_n_ctx(ctx_);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx_);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            RCLCPP_ERROR(get_logger(), "Context size exceeded");
            break;
        }
        
        // Check if the goal has been cancelled
        if (goal_handle->is_canceling()) {
            RCLCPP_INFO(get_logger(), "Goal cancelled");
            result->full_response = "Proceso cancelado";
            result->completed = false;
            goal_handle->canceled(result);
            return;
        }
        
        if (llama_decode(ctx_, batch)) {
            RCLCPP_ERROR(get_logger(), "Failed to decode");
            break;
        }
        
        // Sample the next token
        new_token_id = llama_sampler_sample(sampler_, ctx_, -1);
        
        // Is it an end of generation?
        if (llama_vocab_is_eog(vocab_, new_token_id)) {
            break;
        }
        
        // Convert the token to a string
        char token_buf[256];
        int n = llama_token_to_piece(vocab_, new_token_id, token_buf, sizeof(token_buf), 0, true);
        if (n < 0) {
            RCLCPP_ERROR(get_logger(), "Failed to convert token to piece");
            break;
        }
        
        std::string piece(token_buf, n);
        buffer += piece;
        full_response += piece;
        
        // Check if we should publish feedback
        if (!buffer.empty() && (piece.back() == ' ' || piece.back() == '.' || 
                               piece.back() == ',' || piece.back() == '!' || 
                               piece.back() == '?')) {
            std::vector<std::string> clean_phrases = TextProcessor::clean_text(buffer);
            for (const auto& phrase : clean_phrases) {
                if (!phrase.empty()) {
                    feedback->current_chunk = phrase;
                    RCLCPP_INFO(get_logger(), "Chunk procesado: %s", phrase.c_str());
                    feedback->progress = 0.0; // Normalized progress
                    goal_handle->publish_feedback(feedback);
                }
            }
            buffer.clear();
        }
        
        // Prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
        token_count++;
    }
    
    RCLCPP_INFO(get_logger(), "Toda la respuesta generada por el modelo");
    feedback->current_chunk = "[END_FINAL]";
    RCLCPP_INFO(get_logger(), "PARTE FINAL: %s", feedback->current_chunk.c_str());
    feedback->progress = 1.0;
    goal_handle->publish_feedback(feedback);
    
    // Add response to conversation history
    conversation_history_.push_back({"assistant", full_response});
    
    // Set the result
    result->full_response = full_response;
    result->completed = true;
    goal_handle->succeed(result);
}

} // namespace buddy_chat

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    rclcpp::executors::SingleThreadedExecutor executor;
    auto node = std::make_shared<buddy_chat::LLMLifecycleNode>();
    
    executor.add_node(node->get_node_base_interface());
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}