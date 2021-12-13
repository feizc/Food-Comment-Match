from transformers import PretrainedConfig 
import copy 
import logging



logger = logging.getLogger(__name__) 


class CommentConfig(PretrainedConfig):
    model_type = "Comment"
    '''
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 49408)
        hidden_size (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` :obj:`"quick_gelu"` are supported.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    '''

    def __init__(
        self,
        vocab_size=13086,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


class FoodConfig(PretrainedConfig): 

    model_type = "Food"
    '''
    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (:obj:`int`, `optional`, defaults to 224):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` :obj:`"quick_gelu"` are supported.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    '''

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class FoodCommentConfig(PretrainedConfig):
    model_type = "FoodComment"
    def __init__(
        self,
        comment_config_dict=None,
        food_config_dict=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs
    ):
        super().__init__(comment_config_dict=comment_config_dict, food_config_dict=food_config_dict, **kwargs)

        if comment_config_dict is None:
            comment_config_dict = {}
            logger.info("comment_config_dict is None. Initializing the CommentConfig with default values.")

        if food_config_dict is None:
            food_config_dict = {}
            logger.info("food_config_dict is None. initializing the FoodConfig with default values.")

        self.comment_config = CommentConfig(**comment_config_dict)
        self.food_config = FoodConfig(**food_config_dict)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_comment_food_configs(cls, comment_config: CommentConfig, food_config: FoodConfig, **kwargs):
        r"""
        Instantiate a :class:`FoodCommentConfig` (or a derived class) from comment model configuration and
        food model configuration.
        Returns:
            :class:`CommentFoodConfig`: An instance of a configuration object
        """

        return cls(comment_config_dict=comment_config.to_dict(), food_config_dict=food_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.
        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["comment_config"] = self.comment_config.to_dict()
        output["food_config"] = self.food_config.to_dict()
        return output 

if __name__ == '__main__': 

    food_config = FoodConfig() 
    comment_config = CommentConfig()
    configure = FoodCommentConfig(comment_config_dict=comment_config.to_dict(), \
        food_config_dict=food_config.to_dict()) 
    print(configure) 
