from src.utils.db_manager import DbManager





if __name__ == '__main__':

    path = "/data/fyp23-dlltest/Hong/fyp_dl_library_testing/data/dummy.db"
    db = DbManager(path)
    db.register_model("dummy_dataset", 10)




