from util.path_abstract import PathAbstract

class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/home/jaeyeop/DAVIS17_test'

    @staticmethod
    def db_offline_train_root_dir():
        return '/home/jaeyeop/DAVIS17'

    @staticmethod
    def save_root_dir():
        return './models17_DAVIS17_test'

    @staticmethod
    def save_offline_root_dir():
        return './offline_save_dir'

    @staticmethod
    def models_dir():
        return "/home/jaeyeop/models"
