from typing import List
import sqlite3


class DbManager(object):

    def __init__(self, db_path: str):
        super().__init__()
        self.__conn = sqlite3.connect(db_path, check_same_thread=False)

    def register_model(self, dataset_name: str, node_num: int):
        """
        Register model into model table, return the model id
        """
        INSERT_A_MODEL = '''insert into model(dataset_name, node_num)
                            values(?, ?)
                         '''
        self.__conn.execute(INSERT_A_MODEL, (dataset_name, node_num,))
        self.__conn.commit()

        FETCH_ROWID = '''select last_insert_rowid() from model'''
        cur = self.__conn.cursor()
        cur.execute(FETCH_ROWID)
        return cur.fetchone()[0]


    def update_model_generate_fail_backends(self, model_id: int, fail_backends: List[str]):
        """
        Record fail_backends of the model
        """
        UPDATE_GENERATE_FAIL_BKS = '''update model
                                      set generate_fail_backends = ?
                                      where rowid = ?
                                   '''
        self.__conn.execute(UPDATE_GENERATE_FAIL_BKS, (str(fail_backends), model_id,))
        self.__conn.commit()

    def update_model_crash_backends(self, model_id: int, crash_backends: List[str]):
        """
        Record crash_backends of the model
        """
        UPDATE_CRASH_BKS = '''update model
                              set crash_backends = ?
                              where rowid = ?
                           '''
        self.__conn.execute(UPDATE_CRASH_BKS, (str(crash_backends), model_id,))
        self.__conn.commit()

    def update_model_nan_backends(self, model_id: int, nan_backends: List[str]):
        """
        Record nan_backends of the model
        """
        UPDATE_NAN_BKS = '''update model
                            set nan_backends = ?
                            where rowid = ?
                         '''
        self.__conn.execute(UPDATE_NAN_BKS, (str(nan_backends), model_id,))
        self.__conn.commit()

    def update_model_inf_backends(self, model_id: int, inf_backends: List[str]):
        """
        Record inf_backends of the model
        """
        UPDATE_INF_BKS = '''update model
                            set inf_backends = ?
                            where rowid = ?
                          '''
        self.__conn.execute(UPDATE_INF_BKS, (str(inf_backends), model_id,))
        self.__conn.commit()

    def add_inconsistencies(self, incons: List[tuple]):
        INSERT_INCONS = '''insert into inconsistency(model_id, input_index, output_distance)
                           values(?, ?, ?)
                        '''
        self.__conn.executemany(INSERT_INCONS, incons)
        self.__conn.commit()



    def get_incons_inputs_by_model_id(self, model_id: int, threshlod: float):
        SELECT_INCONS_INPUTS_AND_BKS_BY_MODEL_ID = '''select distinct input_index
                                                      from inconsistency
                                                      where model_id = ? and (output_distance > ? or output_distance is null)
                                                      order by input_index asc
                                                   '''
        cur = self.__conn.cursor()
        cur.execute(SELECT_INCONS_INPUTS_AND_BKS_BY_MODEL_ID, (model_id, threshlod))
        return [res[0] for res in cur.fetchall()]

    def get_huge_incons(self, threshold: float):
        GET_HUGE_INCONS = '''select model_id, backend_pair
                             from inconsistency
                             where model_output_delta > ?
                          '''
        cur = self.__conn.cursor()
        cur.execute(GET_HUGE_INCONS, (threshold,))
        return cur.fetchall()
    
    def get_localization_map(self, incons_id):
        GET_LOCALIZATION_MAP = '''select *
                                  from localization_map
                                  where inconsistency_id == ?
                               '''
        cur = self.__conn.cursor()
        cur.execute(GET_LOCALIZATION_MAP, (incons_id,))
        return cur.fetchall()

    def add_training_incons(self, model_id, backend_pair, model_output_delta):
        INSERT_INCONS = '''insert into inconsistency(model_id, backend_pair, model_output_delta)
                           values(?, ?, ?)
                        '''
        self.__conn.execute(INSERT_INCONS, (model_id, backend_pair, model_output_delta,))
        self.__conn.commit()

        FETCH_ROWID = '''select last_insert_rowid() from inconsistency'''
        cur = self.__conn.cursor()
        cur.execute(FETCH_ROWID)
        return cur.fetchone()[0]
    
 
    def record_status(self, model_id: int, status: list):
        UPDATE_STATUS = '''update model
                           set status = ?
                           where rowid = ?
                        '''
        self.__conn.execute(UPDATE_STATUS, (str(status), model_id,))
        self.__conn.commit()

    def get_model_info(self, model_id: int):
        GET_MODEL_INFO = '''select *
                            from model
                            where rowid = ?
                         '''
        cur = self.__conn.cursor()
        cur.execute(GET_MODEL_INFO, (model_id,))
        return cur.fetchone()

 
    def add_localization_map(self, infos):
        INSERT_LOCALIZATION_MAP = '''insert into localization_map(incons_id, layer_name, outputs_delta, outputs_R, gradients_delta, gradients_R, inbound_layers)
                               values(?, ?, ?, ?, ?, ?, ?)
                            '''
        self.__conn.executemany(INSERT_LOCALIZATION_MAP, infos)
        self.__conn.commit()