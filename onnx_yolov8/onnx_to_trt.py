import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse model file
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None

    # The actual yolov3 or yolov3-tiny must set FP32 mode, precision mode must be FP32
    builder.fp16_mode = False
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    config.flags = 0

    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)

    print('Completed parsing of ONNX file')
    print('Building an engine; this would take a while...')
    engine = builder.build_cuda_engine(network)
    print("Completed creating Engine")

    with open("yolov8n_best.trt", "wb") as f:
        f.write(engine.serialize())
    return engine


if __name__ == "__main__":
    onnx_file_path = '../models/yolov8n_best.onnx'
    engine = build_engine(onnx_file_path)
