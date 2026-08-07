from dimsim.compute.jobs import make_job_id
from dimsim.configs.liquid import BulkLiquid


class TestMakeJobIDs:
    def test_replicate_index_unique(self, liquid_config):
        """Test that when compute_config['replicate_index'] is different, the job_id is different."""
        job_id_1 = make_job_id(liquid_config)
        clone = BulkLiquid(**liquid_config)

        clone["replicate_index"] = 1
        job_id_2 = make_job_id(clone)

        assert job_id_1 != job_id_2

    def test_force_field_unique(self, liquid_config):
        """Test that when compute_config['force_field'] is different, the job_id is different."""
        job_id_1 = make_job_id(liquid_config)
        clone = BulkLiquid(**liquid_config)

        clone["force_field"] = "ff3"
        job_id_2 = make_job_id(clone)

        assert job_id_1 != job_id_2

    def test_n_molecules_unique(self, liquid_config):
        """Test that when compute_config['n_molecules'] is different, the job_id is different."""
        job_id_1 = make_job_id(liquid_config)
        clone = BulkLiquid(**liquid_config)

        clone["n_molecules"] += 1
        job_id_2 = make_job_id(clone)

        assert job_id_1 != job_id_2

    def test_smiles_unique(self, liquid_config):
        """Test that when compute_config['smiles'] is different, the job_id is different."""
        job_id_1 = make_job_id(liquid_config)
        clone = BulkLiquid(**liquid_config)

        clone["smiles"] = ["CCN"]
        job_id_2 = make_job_id(clone)

        assert job_id_1 != job_id_2

    def test_mole_fraction_unique(self, liquid_config):
        """Test that when compute_config['x'] is different, the job_id is different."""
        liquid_config["smiles"] = ["CCO", "O"]
        liquid_config["x"] = [0.25, 0.75]

        job_id_1 = make_job_id(liquid_config)
        clone = BulkLiquid(**liquid_config)

        clone["x"] = [0.5, 0.5]
        job_id_2 = make_job_id(clone)

        assert job_id_1 != job_id_2

    def test_temperature_unique(self, liquid_config):
        """Test that when compute_config['temperature'] is different, the job_id is different."""
        job_id_1 = make_job_id(liquid_config)
        clone = BulkLiquid(**liquid_config)

        clone["temperature"] += 10.0
        job_id_2 = make_job_id(clone)

        assert job_id_1 != job_id_2

    def test_pressure_unique(self, liquid_config):
        """Test that when compute_config['pressure'] is different, the job_id is different."""
        job_id_1 = make_job_id(liquid_config)
        clone = BulkLiquid(**liquid_config)

        clone["pressure"] += 10.0
        job_id_2 = make_job_id(clone)

        assert job_id_1 != job_id_2
