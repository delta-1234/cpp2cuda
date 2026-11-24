"""Exercise specialist handoff and selection without a CUDA device or an API key."""
import importlib.metadata
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from optimize import decision, optimize
from port_agent import Agent, ROOT
from port_task import Task
from profile_kernels import entry_hashes


def metrics(key, simt, tile, end=10.0):
    profile = {m:{"median_us":value,"entry_sass_sha256":[m],"launch_config_sha256":"launch"} for m,value in
               (("simt",simt),("tile",tile))}
    if simt<100: profile["simt"]["entry_sass_sha256"] = ["updated-simt"]
    if tile>=120: profile["tile"]["entry_sass_sha256"] = ["updated-tile"]
    return {"checkpoint":key,"profile":profile,
            "end_to_end_ms":{"many":{"cpu":5.0,"simt":end,"tile":end}}}


class OptimizationTests(unittest.TestCase):
    def test_react_tool_observations_and_retry(self):
        try:
            compatible = importlib.metadata.version("transformers") == "4.41.2"
        except importlib.metadata.PackageNotFoundError:
            compatible = False
        if not compatible: self.skipTest("requires the offline Transformers 4.41.2 runtime")
        from dialog import transformers_dialog
        from port_agent import decode
        tools, errors = [], []
        def model(step, system, memory):
            if step == 1: return "missing tool"
            if step == 2: return '{"tool":"probe","topic":"gpu"}'
            self.assertTrue(any("device-86" in item["content"] for item in memory))
            return '{"tool":"finish"}'
        def dispatch(step, actions):
            tools.extend(a["tool"] for a in actions)
            return "device-86"
        transformers_dialog("test",{},3,model,decode,dispatch,lambda:"finish" in tools,
                            lambda step,error:errors.append(step))
        self.assertEqual(tools,["probe","finish"])
        self.assertEqual(errors,[1])

    def test_missing_profiler_preserves_verified_port(self):
        run = Namespace(args=Namespace(candidate=None), timed=True, env={"cuobjdump":"cuobjdump"},
                        audit={"benchmark":{"many":{}},"validation":{"case":"passed"}})
        optimize(run)
        self.assertEqual(run.audit["missing_tools"],["nsys"])
        self.assertEqual(run.audit["status"],"blocked")
        self.assertIn("validation",run.audit)

    def test_gates(self):
        before = metrics("start",100,100)
        for after, expected in [(metrics("gain",96,101), True), (metrics("noise",99,100),False),
                                (metrics("peer",90,106),False), (metrics("e2e",90,100,11),False)]:
            with self.subTest(key=after["checkpoint"]):
                self.assertEqual(decision("simt",before,after)["accepted"],expected)

    def test_fingerprint_uses_instruction_bytes(self):
        def fingerprint(name, second):
            dump = f"Function : {name}\n /*0000*/ MOV R1, R2; /* 0x1234 */\n /* 0x{second} */\n"
            with patch("profile_kernels.subprocess.run",return_value=Namespace(stdout=dump)):
                return entry_hashes("cuobjdump","replay",{"simt":[name]})["simt"]
        self.assertEqual(fingerprint("kernel_1","aa"),fingerprint("kernel_2","aa"))
        self.assertNotEqual(fingerprint("kernel_1","aa"),fingerprint("kernel_1","ab"))

    def test_prevent_cross_backend_edits(self):
        before, after = metrics("base",100,100), metrics("candidate",60,98)
        after["profile"]["tile"]["entry_sass_sha256"] = ["different-tile"]
        result = decision("simt",before,after)
        self.assertFalse(result["accepted"] or result["peer_preserved"])

    def test_restore_and_independent_handoff(self):
        task = Task(None,ROOT/"examples/window_energy/task.json",[])
        args = Namespace(inspect=False,candidate=None,profile_case="many",specialist_steps=3,specialist_trials=1,task="")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"workflow").mkdir()
            for name in ("probe.py","build_port.sh"):
                (root/"workflow"/name).touch()
            with patch("port_agent.ROOT", root):
                run = Agent(args,{"model":"test"},task)
            code = run.files/"kernel.cu"
            code.write_text("seed")
            run.env.update(driver={"devices":[{"cc":"86"}]},nvcc_version="test",nsys="nsys",cuobjdump="cuobjdump")
            original_act = run.act

            def act(action):
                if action["tool"]=="build":
                    run.audit["build"] = {"cuda_std":14,"host_std":17}
                elif action["tool"]=="verify":
                    run.audit["last_verified_checkpoint"] = run.repair.checkpoint()
                elif action["tool"]=="benchmark":
                    run.timed = True
                    run.audit["benchmark"] = {"many":{"median_ms":{"cpu":5.0,"simt":10.0,"tile":10.0}}}
                else:
                    return original_act(action)

            def measure(agent,case):
                key = run.audit["last_verified_checkpoint"]
                timings = {"seed":(100,100),"both changed":(80,120),"simt improved":(80,100),"tile regression":(80,120)}
                return metrics(key,*timings[code.read_text()])

            def dialog(role,system,initial,turns,dispatch,done):
                self.assertEqual(initial["role"],role)
                self.assertFalse(done())
                self.assertEqual(code.read_text(),"seed" if role=="simt" else "simt improved")
                if role=="simt":
                    dispatch({"tool":"write","path":code.name,"code":"both changed"})
                    with self.assertRaisesRegex(ValueError,"scope check failed"):
                        dispatch({"tool":"trial","hypothesis":"mixed edit"})
                    self.assertFalse(done())
                change = "simt improved" if role=="simt" else "tile regression"
                dispatch({"tool":"write","path":code.name,"code":change})
                result = dispatch({"tool":"trial","hypothesis":"block geometry"})
                self.assertEqual(result["accepted"],role=="simt")
                dispatch({"tool":"finish","summary":"measured"})
                self.assertTrue(done())

            run.act, run.dialog = act, dialog
            for tool in ("build","verify","benchmark"): act({"tool":tool})
            with patch("optimize.measure",side_effect=measure):
                optimize(run)
            self.assertEqual(code.read_text(),"simt improved")
            self.assertEqual(run.audit["status"],"verified")
            self.assertEqual(run.audit["optimization"]["selected"]["checkpoint"],run.repair.checkpoint())


if __name__=="__main__":
    unittest.main()
