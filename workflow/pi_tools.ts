// Pi RPC extension. The Python controller owns the CUDA workspace and executes every operation.
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
    const str = Type.String();
    const integer = Type.Integer();
    const opt = Type.Optional;
    const tools = {
        probe: ["Inspect the GPU, toolchain, system or runtimes.", {topic: Type.Union(["gpu", "toolchain", "system", "runtimes"].map(x => Type.Literal(x)))}],
        select_toolchain: ["Select a probed toolkit and optional host compiler.", {toolkit: str, ccbin: opt(str)}],
        catalog: ["List offline CUDA manuals.", {}],
        search: ["Search offline manual pages by keywords.", {query: str, doc: opt(str)}],
        page: ["Read an offline manual page.", {doc: str, page: integer, offset: opt(integer)}],
        read: ["Read a reference or candidate file from the inventory.", {path: str, start: opt(integer)}],
        experiment: ["Compile and run a small disposable CUDA program.", {code: str, cuda_std: opt(integer)}],
        diagnostics: ["Inspect failures, available logs and verified checkpoints.", {}],
        log: ["Read a recorded tool log by its exact name.", {name: str, offset: opt(integer)}],
        finish: ["Submit the evidence-based summary for controller acceptance.", {summary: str}],
        blocked: ["Report a missing dependency after hardware probes.", {summary: str}],
        case: ["Propose a reproducible CPU test with command-line arguments.", {name: str, args: Type.Array(str), input: opt(str)}],
        baseline: ["Build the independent CPU oracle and collect expected outputs.", {}],
        write: ["Write a complete candidate source file.", {path: str, code: str}],
        edit: ["Replace a unique substring in a candidate.", {path: str, old: str, new: str}],
        remove: ["Remove a candidate source file.", {path: str}],
        build: ["Compile the current candidate.", {cuda_std: opt(integer), host_std: opt(integer)}],
        verify: ["Validate both GPU paths against the CPU and trace their launches.", {}],
        benchmark: ["Measure the fully verified revision.", {}],
        test_case: ["Run one named CPU-oracle case on one built backend.", {case: str, mode: Type.Union([Type.Literal("simt"), Type.Literal("tile")])}],
        restore: ["Restore the files of a verified checkpoint.", {checkpoint: str}],
        trial: ["Build, verify, profile and compare this specialist's change.", {hypothesis: str}],
        reset_best: ["Restore and revalidate the specialist's incumbent.", {}],
    } as const;

    for (const name of process.env.FACTORPORT_PI_TOOLS!.split(",")) {
        const [description, fields] = tools[name as keyof typeof tools];
        pi.registerTool({
            name: "fp_" + name, label: name, description,
            parameters: Type.Object(fields, {additionalProperties: false}),
            executionMode: "sequential",
            async execute(_id, params, _signal, _update, ctx) {
                const reply = await ctx.ui.input("factorport.call", JSON.stringify({params, name}));
                if (!reply) throw new Error("FactorPort controller disconnected");
                if (JSON.parse(reply).error) throw new Error(reply);
                return {content: [{type: "text", text: reply}], details: {}};
            },
        });
    }
    // Stop in Pi before it can submit the automatic request following the last tool result.
    let turns = 0;
    const budget = Number(process.env.FACTORPORT_PI_TURNS);
    pi.on("turn_end", (_event, ctx) => {
        if (++turns >= budget) ctx.abort();
    });
    pi.on("session_start", (_event, ctx) => {
        ctx.ui.notify("factorport.ready " + JSON.stringify(pi.getActiveTools()), "info");
    });
}
