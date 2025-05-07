#include "llm_lifecycle_node.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

namespace coco_chat {

std::vector<std::string> TextProcessor::clean_text(const std::string& text) {
    std::locale::global(std::locale(""));
    std::cout.imbue(std::locale());
        
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wtext = converter.from_bytes(text);
    
    std::wregex punct_regex(L"[!¡?¿*,.:;()\\[\\]{}]");
    std::wstring wcleaned = std::regex_replace(wtext, punct_regex, L" ");
    
    std::wregex space_regex(L"\\s+");
    wcleaned = std::regex_replace(wcleaned, space_regex, L" ");
    
    std::string cleaned = converter.to_bytes(wcleaned);
        
    std::vector<std::string> sentences;
    std::regex sentence_regex("([.!?])\\s+");
    std::sregex_token_iterator iter(cleaned.begin(), cleaned.end(), sentence_regex, {-1, 0});
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
    
    std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("coco_chat");
    model_path_ = pkg_share_dir + "/models/LLM/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8/meta-llama-3.1-8b-instruct-q4_0.gguf";
    
    RCLCPP_INFO(get_logger(), "Model path: %s", model_path_.c_str());
}

LLMLifecycleNode::~LLMLifecycleNode() {
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
        ggml_backend_load_all();
        
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 12;  
        
        model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
        if (!model_) {
            RCLCPP_ERROR(get_logger(), "Error: unable to load model");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
        }
        
        vocab_ = llama_model_get_vocab(model_);
        
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 4096;
        ctx_params.n_batch = 2048;
        ctx_params.n_threads = 4;
        
        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            RCLCPP_ERROR(get_logger(), "Error: failed to create llama_context");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
        }
        
        sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler_, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.5f)); 
        llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
                        
        action_server_ = rclcpp_action::create_server<coco_interfaces::action::ProcessResponse>(
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
        
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LLMLifecycleNode::on_deactivate(const rclcpp_lifecycle::State& state) {
    RCLCPP_INFO(get_logger(), "Deactivating LLM Node");
        
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_action::GoalResponse LLMLifecycleNode::handle_goal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const coco_interfaces::action::ProcessResponse::Goal> goal)
{
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse LLMLifecycleNode::handle_cancel(
    const std::shared_ptr<GoalHandleProcessResponse> goal_handle)
{
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
}

void LLMLifecycleNode::handle_accepted(const std::shared_ptr<GoalHandleProcessResponse> goal_handle)
{
    std::thread{std::bind(&LLMLifecycleNode::execute_response_generation, this, _1), goal_handle}.detach();
}

void LLMLifecycleNode::execute_response_generation(const std::shared_ptr<GoalHandleProcessResponse> goal_handle)
{
    std::lock_guard<std::mutex> lock(llama_mutex_); 
    
    auto feedback = std::make_shared<coco_interfaces::action::ProcessResponse::Feedback>();
    auto result = std::make_shared<coco_interfaces::action::ProcessResponse::Result>();

    if (!model_ || !ctx_ || !sampler_) {
        result->completed = false;
        goal_handle->abort(result);
        return;
    }

    const std::string system_prompt = 
        "Eres Coco, un asistente virtual amigable para niños que se encuentran en hospitales "
        "y tienen entre 7 a 12 años.";
    
    conversation_history_.clear();
    conversation_history_.push_back({"system", system_prompt});
    conversation_history_.push_back({"user", goal_handle->get_goal()->input_text});
    
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
    RCLCPP_INFO(get_logger(), "Mensaje de entrada: %s", goal_handle->get_goal()->input_text.c_str());
    
    const int n_prompt_tokens = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), llama_get_kv_cache_used_cells(ctx_) == 0, true) < 0) {
        RCLCPP_ERROR(get_logger(), "Failed to tokenize the prompt");
        result->full_response = "Error: failed to tokenize prompt";
        result->completed = false;
        goal_handle->abort(result);
        return;
    }
    
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    std::string buffer;
    std::string full_response;
    
    llama_token new_token_id;
    int max_tokens = 300; 
    int token_count = 0;
    
    RCLCPP_INFO(get_logger(), "Modelo: ");
    while (token_count < max_tokens) {
        int n_ctx = llama_n_ctx(ctx_);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx_);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            RCLCPP_ERROR(get_logger(), "Context size exceeded");
            break;
        }
        
        if (goal_handle->is_canceling()) {
            result->full_response = "Proceso cancelado";
            result->completed = false;
            goal_handle->canceled(result);
            return;
        }
        
        if (llama_decode(ctx_, batch)) {
            RCLCPP_ERROR(get_logger(), "Failed to decode");
            break;
        }
        
        new_token_id = llama_sampler_sample(sampler_, ctx_, -1);
        
        if (llama_vocab_is_eog(vocab_, new_token_id)) {
            break;
        }
        
        char token_buf[256];
        int n = llama_token_to_piece(vocab_, new_token_id, token_buf, sizeof(token_buf), 0, true);
        if (n < 0) {
            RCLCPP_ERROR(get_logger(), "Failed to convert token to piece");
            break;
        }
        
        std::string piece(token_buf, n);
        buffer += piece;
        full_response += piece;
        
        size_t sentence_end = 0;
        size_t buffer_len = buffer.length();

        while (sentence_end < buffer_len) {
            size_t next_end = buffer.find_first_of(".!?", sentence_end);
            
            if (next_end == std::string::npos) break;
            
            if (next_end + 1 > buffer_len || std::isspace(buffer[next_end + 1])) {
                sentence_end = next_end + 1;
            } else {
                sentence_end = next_end + 1;
                continue;
            }
            
            std::string sentence = buffer.substr(0, sentence_end);
            sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);  
            
            if (!sentence.empty()) {
                std::string clean_phrase = TextProcessor::clean_text(sentence).front();
                feedback->current_chunk = clean_phrase;
                feedback->progress = 0.0;
                std::cout << clean_phrase << std::endl;
                goal_handle->publish_feedback(feedback);
                
                buffer = buffer.substr(sentence_end);
                buffer_len = buffer.length();
                sentence_end = 0;  
            }
        }
        
        batch = llama_batch_get_one(&new_token_id, 1);
        token_count++;
    }
    
    feedback->current_chunk = "[END_FINAL]";
    feedback->progress = 1.0;
    goal_handle->publish_feedback(feedback);
    
    conversation_history_.push_back({"assistant", full_response});
    
    result->full_response = full_response;
    result->completed = true;
    goal_handle->succeed(result);
}

}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    rclcpp::executors::SingleThreadedExecutor executor;
    auto node = std::make_shared<coco_chat::LLMLifecycleNode>();
    
    executor.add_node(node->get_node_base_interface());
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}