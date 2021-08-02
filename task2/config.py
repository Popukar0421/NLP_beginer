from configparser import ConfigParser


class ModelConfig:
    def __init__(self, model_type):
        self.epochs_num = None
        cfg = ConfigParser()
        cfg.read("config.ini", encoding='utf-8')
        model_name = "model_task2"
        self.model_type = model_type
        self.name = cfg.get(model_name, "name")
        self.model_checkpoints_path = cfg.get(model_name, 'model_checkpoints_path')
        # self.model_pb_path = cfg.get(model_name, 'model_pb_path')
        self.train_path = cfg.get(model_name, 'train_path')
        # self.val_path = cfg.get(model_name, 'val_path')
        self.vocab_path = cfg.get(model_name, 'vocab_path')
        # self.labels = cfg.get(model_name, 'labels')
        self.embedding_dim = cfg.get(model_name, 'embedding_dim')
        self.max_seq_length = cfg.getint(model_name, 'max_seq_length')
        self.num_classes = cfg.getint(model_name, 'num_classes')
        # self.num_filters = cfg.getint(model_name, 'num_filters')
        # self.kernel_size = cfg.getint(model_name, 'kernel_size')
        # self.vocab_size = cfg.getint(model_name, 'vocab_size')
        self.hidden_dim = cfg.getint(model_name, 'hidden_dim')
        self.keep_prob = cfg.getfloat(model_name, 'keep_prob')
        self.learning_rate = cfg.getfloat(model_name, 'learning_rate')
        self.batch_size = cfg.getint(model_name, 'batch_size')
        self.num_epochs = cfg.getint(model_name, 'num_epochs')
        self.print_per_batch = cfg.getint(model_name, 'print_per_batch')
        self.save_per_batch = cfg.getint(model_name, 'save_per_batch')
        self.is_l2 = cfg.getboolean(model_name, 'is_l2')




