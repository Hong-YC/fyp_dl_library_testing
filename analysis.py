from utils.db_manager import DbManager
from pathlib import Path


db_manager = DbManager(str(Path.cwd() / 'data' / 'dummy.db'))

# print(db_manager.get_model_info(3))
print(db_manager.get_huge_incons(1e-4))