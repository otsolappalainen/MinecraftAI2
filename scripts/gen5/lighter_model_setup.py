class FullModelFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # Initialize with a placeholder for features_dim; it will be overwritten
        super(FullModelFeatureExtractor, self).__init__(observation_space, features_dim=256)

        # Scalar observation processing
        scalar_input_dim = observation_space["other"].shape[0]
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        scalar_output_size = 128

        # Lighter CNN layers for image processing
        image_input_channels = observation_space["image"].shape[0]  # Assuming 3 for RGB images
        self.image_net = nn.Sequential(
            nn.Conv2d(image_input_channels, 16, kernel_size=6, stride=3),  # Reduced filters and adjusted kernel
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # Reduced filters
            nn.ReLU(),
            nn.Flatten(),
        )
        # Precompute the CNN output size
        dummy_image = th.zeros(1, *observation_space["image"].shape)
        conv_output_size = self._get_conv_output_size(dummy_image)
        self.conv_output_size = conv_output_size  # Store for debugging

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(scalar_output_size + conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self._features_dim = 128

        print("Using lighter CNN layers for feature extractor.")
