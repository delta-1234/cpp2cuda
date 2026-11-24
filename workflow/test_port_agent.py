"""Portable regressions for the agent tool protocol, offline library and CPU contract."""
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from api_agent import OfflineRun
from knowledge import Library
from port_agent import Agent, ROOT, decode
from port_task import CAPTURE, Task
from repair import RepairHistory


GPU = "--gpu" in sys.argv
if GPU: sys.argv.remove("--gpu")


class PortTests(unittest.TestCase):
    def test_interrupted_stage_reaps_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = OfflineRun(Path(tmp), {}, Namespace(stage_timeout=30), "stop-test-")
            with patch("api_agent.subprocess.Popen") as spawn, patch("api_agent.os.killpg") as kill:
                proc = spawn.return_value
                proc.returncode = None
                proc.wait.side_effect = [KeyboardInterrupt(), -signal.SIGKILL]
                with self.assertRaises(KeyboardInterrupt):
                    run.offline("build", ["unused"])
                self.assertTrue(spawn.call_args.kwargs["start_new_session"])
                kill.assert_called_once_with(proc.pid, signal.SIGKILL)
                self.assertEqual(proc.wait.call_count, 2)

    def test_knowledge_and_protocol(self):
        lib = Library()
        hit = lib.search("__syncthreads", "cuda-guide-11.3")[0]
        self.assertEqual(hit["version"], "11.3")
        self.assertIn("__syncthreads", lib.pages(hit["doc"])[hit["page"]-1])
        with self.assertRaises(ValueError):
            lib.read(hit["doc"],0)
        self.assertEqual(decode('<<<write main.cu>>>\na\n<<<END>>>')[0]["code"],"a\n")
        self.assertEqual(decode('[{"tool":"probe","topic":"gpu"}]')[0]["topic"],"gpu")

    def test_manifest_and_proposed_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"lib.hpp").write_text("#pragma once")
            main = root/"input.cpp"
            main.write_text('#include "lib.hpp"\n')
            (root/".env").write_text("private")
            spec = root/"task.json"
            spec.write_text(json.dumps({"source":main.name,"cases":[{"name":"base","args":["7"]}]}))
            task = Task(None,spec,[])
            self.assertEqual(set(task.files),{main.name,"lib.hpp"})
            with self.assertRaises(ValueError):
                task.propose("base",[],None)
            task.propose("edge",["{input}","{output}"],"1\n")
            self.assertEqual(task.cases["edge"],["agent-inputs/edge.txt","{output}"])
            with self.assertRaises(ValueError):
                Task(main,None,[".env"])
            self.assertFalse(Task(main,None,[]).cases)

    def test_factor_profile_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"renamed.cpp"
            source.write_text('#include "factor_replay.hpp"\n')
            (root/"factor_replay.hpp").write_text("original")
            spec = root/"spec.json"
            spec.write_text(json.dumps({"profile":"factor","source":source.name}))
            task = Task(None,spec,[])
            self.assertTrue(task.factor)
            task.snapshot(root/"snapshot")
            (root/"factor_replay.hpp").write_text("modified")
            self.assertEqual((task.root/"factor_replay.hpp").read_text(),"original")
            self.assertIsNone(Task(None,None,[],True).entry)

    def test_cpu_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = str(Path(tmp)/"output")
            def capture(cmd):
                return subprocess.run([sys.executable,"-c",CAPTURE,name,*cmd],capture_output=True)
            self.assertEqual(capture([sys.executable,"-c","print(42)"]).returncode,0)
            self.assertEqual(Path(name).read_bytes(), b"42\n")
            self.assertEqual(capture([sys.executable,"-c","import sys; open(sys.argv[1],'w').write('7')","{output}"]).returncode,0)
            self.assertEqual(Path(name).read_bytes(), b"7")
            self.assertNotEqual(capture([sys.executable,"-c","pass","{output}"]).returncode,0)

    def test_repeat_detection_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port = root/"work/port"
            port.mkdir(parents=True)
            f = port/"kernel.cu"
            f.write_text("known revision")
            repair = RepairHistory(root,port)
            ckpt = repair.checkpoint()
            f.write_text("regression")
            (port/"extra.cpp").write_text("extra")
            target = {"case":"tail","mode":"tile"}
            repair.failure("verify","bad output",target)
            repeat = repair.failure("verify","bad output",target)
            self.assertEqual(repeat["loop"],"repeated_failure")
            for tool in ("build","verify","build"):
                alternating = repair.failure(tool,"compiler failed" if tool=="build" else "bad output",
                                             None if tool=="build" else target)
            self.assertEqual(alternating["loop"],"alternating_failures")
            repair.passed("test_case",target)
            self.assertEqual(repeat["status"],"open")
            previous = repair.restore(ckpt)
            self.assertEqual((root/previous/f.name).read_text(),"regression")
            self.assertEqual(f.read_text(),"known revision")
            self.assertFalse((port/"extra.cpp").exists())
            repair.passed("verify")
            self.assertEqual(repeat["status"],"passed_on_retry")
            (repair.folder/ckpt/f.name).write_text("tampered")
            with self.assertRaises(ValueError): repair.restore(ckpt)
            self.assertEqual(f.read_text(),"known revision")

    def test_finish_and_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"workflow").mkdir()
            for name in ("build_port.sh","probe.py"):
                shutil.copyfile(ROOT/"workflow"/name,root/"workflow"/name)
            task = Task(None,ROOT/"examples/window_energy/task.json",[])
            with patch("port_agent.ROOT",root):
                agent = Agent(Namespace(inspect=False),{"model":"test"},task)
            agent.audit["observations"] = {"gpu":{},"toolchain":{}}
            with self.assertRaises(ValueError):
                agent.act({"tool":"finish","summary":"ready"})
            agent.built = agent.verified = agent.timed = True
            agent.act({"tool":"write","path":"a.cu","code":"// changed"})
            self.assertFalse(agent.built or agent.verified or agent.timed)
            agent.act({"tool":"edit","path":"a.cu","old":"changed","new":"revised"})
            self.assertEqual((agent.files/"a.cu").read_text(),"// revised")
            agent.act({"tool":"remove","path":"a.cu"})
            self.assertFalse((agent.files/"a.cu").exists())
            with self.assertRaises(ValueError):
                agent.act({"tool":"write","path":"../b.cu","code":"invalid"})
            with self.assertRaises(ValueError):
                agent.act({"tool":"benchmark"})


@unittest.skipUnless(GPU, "use --gpu on the CUDA node")
class GPURepairTests(unittest.TestCase):
    def test_repair_loop_and_turn_limit_rollback(self):
        project = ROOT/"examples/window_energy"
        args = Namespace(inspect=False, stage_timeout=180, max_steps=4, model_timeout=60, task="regression",
                         candidate=project/"repair_fixture",extra_orders=[])
        agent = Agent(args,{"model":"scripted-controller-test"},Task(None,project/"task.json",[]))
        path = "window_energy.cu"
        bad, good = "int g = base - R + idx + (base != 0);", "int g = base - R + idx;"
        fix = {"tool":"edit","path":path,"old":bad,"new":good}
        regress = {"tool":"edit","path":path,"old":good,"new":bad}
        turns = [
            [{"tool":"probe","topic":"gpu"},{"tool":"probe","topic":"toolchain"},{"tool":"build"},{"tool":"verify"}],
            [{"tool":"verify"}],
            [fix,{"tool":"build"},{"tool":"test_case","case":"tail","mode":"tile"},{"tool":"verify"},{"tool":"benchmark"}],
            [regress,{"tool":"build"},{"tool":"verify"}],
        ]
        try:
            with patch("port_agent.complete",side_effect=[json.dumps(t) for t in turns]):
                agent.execute()
        finally:
            agent.record()
        self.assertEqual(agent.audit["status"],"verified_fallback")
        self.assertIn(good,(agent.files/path).read_text())
        draft=agent.dir/agent.audit["fallback"]["previous_draft"]/path
        self.assertIn(bad,draft.read_text())
        self.assertTrue(any(f.get("loop")=="repeated_failure" for f in agent.repair.failures))
        self.assertTrue(agent.audit["focused_retests"])
        self.assertEqual(set(agent.audit["validation"]["successful_gpu_launches"]),{"simt","tile"})
        print("GPU repair/rollback regression:",agent.dir/"report.json")


if __name__ == "__main__":
    unittest.main()
