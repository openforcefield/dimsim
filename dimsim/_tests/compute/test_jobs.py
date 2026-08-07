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
