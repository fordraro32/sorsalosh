class AdapterConfig:
    def __init__(self):
        self.input_dim = 768  # Assuming base model hidden size
        self.bottleneck_dim = 64
        self.num_gpus = 8
        self.max_memory = 2000 * 1024 * 1024 * 1024  # 2000 GB in bytes
        self.max_sequence_length = 10000
        self.supported_languages = ["python", "cpp", "javascript", "java", "ruby", "go"]
        self.use_memory = True
