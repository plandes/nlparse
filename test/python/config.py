from zensols.config import ExtendedInterpolationConfig


class AppConfig(ExtendedInterpolationConfig):
    SECTION = 'test'

    def __init__(self, name='nlparse'):
        super(AppConfig, self).__init__(
            config_file=f'test-resources/{name}.conf', default_expect=True)

    @property
    def parse_path(self):
        return self.get_option_path('parse_test', self.SECTION)

    @property
    def feature_path(self):
        return self.get_option_path('feature_test', self.SECTION)
