import onnxruntime
import os


class PredictBase(object):
    def __init__(self):
        pass

    def get_onnx_session(self, model_dir, use_gpu):
        """
        创建ONNX Runtime推理会话，包含错误处理和兼容性设置
        """
        # 检查模型文件是否存在
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model file not found: {model_dir}")
        
        # 设置会话选项
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        # 禁用某些可能导致问题的优化
        session_options.enable_mem_pattern = False
        session_options.enable_cpu_mem_arena = False
        
        # 设置提供者
        if use_gpu:
            providers = []
            # 尝试使用CUDA
            if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
            providers.append('CPUExecutionProvider')
        else:
            providers = ['CPUExecutionProvider']
        
        print(f"Loading model: {model_dir}")
        print(f"Available providers: {onnxruntime.get_available_providers()}")
        print(f"Using providers: {providers}")
        
        try:
            # 尝试创建推理会话
            onnx_session = onnxruntime.InferenceSession(
                model_dir, 
                session_options, 
                providers=providers
            )
            print(f"Model loaded successfully with providers: {onnx_session.get_providers()}")
            return onnx_session
            
        except Exception as e:
            print(f"Failed to load model with error: {e}")
            # 回退到只使用CPU提供者
            print("Falling back to CPU-only execution...")
            try:
                providers = ['CPUExecutionProvider']
                session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                onnx_session = onnxruntime.InferenceSession(
                    model_dir, 
                    session_options, 
                    providers=providers
                )
                print(f"Model loaded successfully with CPU fallback")
                return onnx_session
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to load model even with CPU fallback: {fallback_error}")

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
