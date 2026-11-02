import os
import sys
import unittest
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.training_pipeline import (
    ingest_and_validate_data,
    train_and_evaluate_model,
    save_and_register_model
)

class TestAirflowPipelineSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.train_path = str(cls.root_dir / "raw_data" / "medical_tc_train.csv")
        cls.test_path = str(cls.root_dir / "raw_data" / "medical_tc_test.csv")
        cls.labels_path = str(cls.root_dir / "raw_data" / "medical_tc_labels.csv")

        cls.temp_model_path = str(cls.root_dir / "models" / "test_pipeline_model.joblib")
        cls.temp_meta_path = str(cls.root_dir / "models" / "test_pipeline_meta.json")

    @classmethod
    def tearDownClass(cls):
        # Limpeza de artefatos de teste temporários
        for p in [cls.temp_model_path, cls.temp_meta_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_task_1_ingest_and_validate(self):
        """Valida a execução da Task 1 (Ingestão e Validação)."""
        info = ingest_and_validate_data(
            train_path=self.train_path,
            labels_path=self.labels_path
        )
        self.assertEqual(info["status"], "validated")
        self.assertGreaterEqual(info["total_samples"], 10000)
        self.assertEqual(len(info["class_distribution"]), 5)

    def test_task_2_and_3_train_and_save(self):
        """Valida a execução das Tasks 2 e 3 (Treinamento, Quality Gate e Persistência)."""
        model, metrics = train_and_evaluate_model(
            train_path=self.train_path,
            test_path=self.test_path,
            labels_path=self.labels_path,
            min_f1_threshold=0.50
        )
        self.assertTrue(metrics["quality_gate_passed"])
        self.assertGreater(metrics["f1_macro"], 0.50)
        self.assertGreater(metrics["accuracy"], 0.50)

        # Task 3: Persistência
        saved_meta = save_and_register_model(
            model=model,
            metrics=metrics,
            model_output_path=self.temp_model_path,
            metadata_output_path=self.temp_meta_path,
            version="9.9.9-test"
        )
        self.assertTrue(os.path.exists(self.temp_model_path))
        self.assertTrue(os.path.exists(self.temp_meta_path))
        self.assertEqual(saved_meta["version"], "9.9.9-test")

    def test_dag_file_syntax_and_importability(self):
        """Verifica se o arquivo da DAG é sintaticamente válido."""
        dag_path = self.root_dir / "dags" / "medical_retrain_dag.py"
        self.assertTrue(dag_path.exists())

        # Teste de compilação Python do arquivo da DAG
        import py_compile
        compiled = py_compile.compile(str(dag_path), doraise=True)
        self.assertTrue(compiled is not None)

if __name__ == "__main__":
    unittest.main()

