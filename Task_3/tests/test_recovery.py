# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check durable checkpoint copies, interrupted publication and corrupt-backup rejection."""

import tempfile
import unittest
from pathlib import Path

from Task_3.cloud.colab import DriveSync


class RecoveryTests(unittest.TestCase):
    def test_verified_restore_predecessor_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, remote = root / 'local', root / 'drive'
            relative = 'checkpoints/SYNTHETIC/recovery.pt'
            source = local / relative
            source.parent.mkdir(parents=True)
            source.write_bytes(b'SYNTHETIC FULL RECOVERY STATE')
            sync = DriveSync(local, remote)
            sync.publish([relative])
            source.unlink()
            self.assertEqual(sync.restore(), 1)
            self.assertEqual(source.read_bytes(), b'SYNTHETIC FULL RECOVERY STATE')
            self.assertEqual(sync.restore(), 0)
            # Simulate interruption after replacement but before index publication.
            target = remote / relative
            previous = target.with_name(target.name + '.previous')
            target.replace(previous)
            target.write_bytes(b'UNCOMMITTED NEW STATE')
            source.unlink()
            self.assertEqual(DriveSync(local, remote).restore(), 1)
            self.assertEqual(source.read_bytes(), b'SYNTHETIC FULL RECOVERY STATE')
            previous.write_bytes(b'CORRUPT PREDECESSOR')
            with self.assertRaises(RuntimeError):
                DriveSync(local, remote).restore()


if __name__ == '__main__':
    unittest.main()
