PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "model" (
                        "dataset_name"	TEXT,
                        "node_num"	INTEGER NOT NULL,
                        "generate_fail_backends"	TEXT,
                        "crash_backends"	TEXT,
                        "nan_backends"	TEXT,
                        "inf_backends"	TEXT,
                        "status"	TEXT
                    );
CREATE TABLE IF NOT EXISTS "inconsistency" (
                        "model_id"	INTEGER NOT NULL,
                        "model_output_delta"	REAL,
                        PRIMARY KEY("model_id")
                    );
CREATE TABLE IF NOT EXISTS "localization_map" (
                    "incons_id"	INTEGER NOT NULL,
                    "layer_name"	TEXT NOT NULL,
                    "outputs_delta"	REAL,
                    "outputs_R"	REAL,
                    "gradients_delta"	REAL,
                    "gradients_R"	REAL,
                    "weights_delta"	REAL,
                    "inbound_layers"	TEXT,
                    PRIMARY KEY("incons_id","layer_name")
                );
COMMIT;
