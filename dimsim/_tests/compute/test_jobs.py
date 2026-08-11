from dimsim.compute.jobs import make_job_id
from dimsim.configs.liquid import BulkLiquid


class TestMakeJobIDs:
    def test_job_id_same_multiple_calls(self, liquid_config):
        """Test that calling make_job_id multiple times with the same compute_config returns the same job_id."""
        job_id_1 = make_job_id(liquid_config)
        job_id_2 = make_job_id(liquid_config)

        assert job_id_1 == job_id_2

    def test_recreate_compute_config_same_job_id(self, liquid_config):
        """Test that creating a new compute_config with the same parameters returns the same job_id."""
        job_id_1 = make_job_id(liquid_config)
        job_id_2 = make_job_id(BulkLiquid(**liquid_config))

        assert job_id_1 == job_id_2

    def test_id_same_as_hardcoded(self):
        other_liquid_config = BulkLiquid(
            tag="liquid",
            force_field="test_force_field",
            n_molecules=400,
            replicate_index=1,
            smiles=["CCO", "O"],
            x=[0.1, 0.9],
            density=0.86,
            temperature=301.11,
            pressure=98.875,
        )

        job_id = make_job_id(other_liquid_config)

        # hard-coded from an interpreter run
        assert job_id == "35210a3d45f082dfb8cd85968cce08f860d2431fad0496c12c9f1627b834ed52"

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
