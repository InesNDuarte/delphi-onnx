#!/usr/bin/env node
import http from "node:http"
import { execFile } from "node:child_process"
import { readFile, stat, mkdir, writeFile } from "node:fs/promises"
import { createReadStream } from "node:fs"
import { createHash } from "node:crypto"
import os from "node:os"
import path from "node:path"
import process from "node:process"
import { fileURLToPath } from "node:url"
import { chromium } from "playwright-core"
import { promisify } from "node:util"

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
const DEFAULTS = {
    backends: ["wasm", "webgpu"],
    coldTrials: 10,
    warmups: 10,
    runs: 200,
    trajectoryRuns: 30,
    manifestUrl: "/evaluation/generated/manifest.json",
    memoryOnly: false,
    fidelityOnly: false,
    latencyOnly: false
}
const execFileAsync = promisify(execFile)

const sha256File = async filename => {
    const hash = createHash("sha256")
    for await (const chunk of createReadStream(filename)) hash.update(chunk)
    return hash.digest("hex")
}

const evaluationProvenance = async (manifest, manifestPath) => ({
    manifest_sha256: await sha256File(manifestPath),
    upstream_commit: manifest.upstream.commit,
    validation_sha256: manifest.upstream.validation_sha256,
    upstream_model_source_sha256: manifest.upstream.model_source_sha256,
    checkpoint_sha256: manifest.checkpoint.sha256,
    onnx_sha256: manifest.onnx.sha256,
    labels_sha256: manifest.labels.sha256,
    sdk_sha256: await sha256File(path.join(ROOT, "delphiSDK.js")),
    all_final_reference_sha256: await sha256File(
        path.join(ROOT, "evaluation", "generated", manifest.all_cohort.reference_file)
    ),
    stratified_full_reference_sha256: await sha256File(
        path.join(ROOT, "evaluation", "generated", manifest.stratified.cases[0].logits_file)
    )
})

const parseArguments = argv => {
    const options = { ...DEFAULTS }
    for (let index = 0; index < argv.length; index++) {
        const argument = argv[index]
        if (argument === "--backend") options.backends = [argv[++index]]
        else if (argument === "--cold-trials") options.coldTrials = Number(argv[++index])
        else if (argument === "--warmups") options.warmups = Number(argv[++index])
        else if (argument === "--runs") options.runs = Number(argv[++index])
        else if (argument === "--trajectory-runs") options.trajectoryRuns = Number(argv[++index])
        else if (argument === "--manifest-url") options.manifestUrl = argv[++index]
        else if (argument === "--memory-only") options.memoryOnly = true
        else if (argument === "--fidelity-only") options.fidelityOnly = true
        else if (argument === "--latency-only") options.latencyOnly = true
        else if (argument === "--help") {
            console.log(`Usage: node evaluation/run_browser_evaluation.mjs [options]

  --backend wasm|webgpu   Run one backend (default: both)
  --cold-trials N         Fresh-Chrome initialization trials (default: 10)
  --warmups N             Warm-up runs per context length (default: 10)
  --runs N                Timed runs per context length (default: 200)
  --trajectory-runs N     Timed 32-event rollouts (default: 30)
  --manifest-url URL      Served manifest URL
  --memory-only           Run only the dedicated process-RSS trial
  --fidelity-only         Refresh fidelity fields in existing result files
  --latency-only          Refresh latency fields in existing result files`)
            process.exit(0)
        } else throw new Error(`Unknown option: ${argument}`)
    }
    for (const backend of options.backends) {
        if (!["wasm", "webgpu"].includes(backend)) throw new Error(`Unsupported backend: ${backend}`)
    }
    return options
}

const contentType = filename => ({
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
    ".f32": "application/octet-stream"
}[path.extname(filename)] || "application/octet-stream")

const startServer = async () => {
    const server = http.createServer(async (request, response) => {
        try {
            const requested = decodeURIComponent(new URL(request.url, "http://localhost").pathname)
            const relative = requested === "/" ? "evaluation/browser/harness.html" : requested.slice(1)
            const filename = path.resolve(ROOT, relative)
            if (filename !== ROOT && !filename.startsWith(`${ROOT}${path.sep}`)) {
                response.writeHead(403).end("Forbidden")
                return
            }
            const information = await stat(filename)
            if (!information.isFile()) throw new Error("Not a file")
            response.writeHead(200, {
                "Content-Type": contentType(filename),
                "Content-Length": information.size,
                "Cache-Control": "no-store",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Cross-Origin-Resource-Policy": "same-origin"
            })
            createReadStream(filename).pipe(response)
        } catch (error) {
            response.writeHead(404, { "Content-Type": "text/plain" }).end("Not found")
        }
    })
    await new Promise((resolve, reject) => {
        server.once("error", reject)
        server.listen(0, "127.0.0.1", resolve)
    })
    const address = server.address()
    return { server, origin: `http://127.0.0.1:${address.port}` }
}

const attachPageLogging = (page, prefix) => {
    page.on("console", message => console.log(`[${prefix}] ${message.text()}`))
    page.on("pageerror", error => console.error(`[${prefix}] PAGE ERROR: ${error.message}`))
}

const loadHarness = async (page, origin, prefix) => {
    attachPageLogging(page, prefix)
    page.setDefaultTimeout(0)
    await page.goto(`${origin}/evaluation/browser/harness.html`, { waitUntil: "load" })
    await page.waitForFunction(() => window.evaluationHarnessReady === true)
}

const launchBrowser = () => chromium.launch({
    executablePath: CHROME,
    headless: true,
    args: [
        "--headless=new",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-extensions",
        "--disable-sync",
        "--no-first-run"
    ]
})

const chromeRss = async excludedPids => {
    const { stdout } = await execFileAsync("/bin/ps", ["-axo", "pid=,rss=,command="])
    let rssKiB = 0
    const pids = []
    for (const line of stdout.split("\n")) {
        const match = line.match(/^\s*(\d+)\s+(\d+)\s+(.+)$/)
        if (!match) continue
        const pid = Number(match[1])
        const command = match[3]
        if (!command.includes("Google Chrome") || excludedPids.has(pid)) continue
        pids.push(pid)
        rssKiB += Number(match[2])
    }
    return { pids, bytes: rssKiB * 1024 }
}

const allChromePids = async () => new Set((await chromeRss(new Set())).pids)

const runMemoryTrial = async (backend, origin, example) => {
    const excludedPids = await allChromePids()
    const browser = await launchBrowser()
    let interval
    try {
        const context = await browser.newContext()
        const page = await context.newPage()
        await loadHarness(page, origin, `${backend}:memory`)
        const idle = await chromeRss(excludedPids)
        let peakBytes = idle.bytes
        let samples = 1
        interval = setInterval(() => {
            chromeRss(excludedPids).then(sample => {
                peakBytes = Math.max(peakBytes, sample.bytes)
                samples++
            }).catch(() => {})
        }, 100)
        const inPage = await page.evaluate(
            configuration => window.runMemoryWorkload(configuration),
            { backend, tokens: example.tokens.slice(-48), ages: example.ages_days.slice(-48) }
        )
        clearInterval(interval)
        interval = undefined
        const final = await chromeRss(excludedPids)
        peakBytes = Math.max(peakBytes, final.bytes)
        await context.close()
        return {
            method: "Peak summed RSS of Chrome processes newly launched for the dedicated trial, sampled every 100 ms",
            caveat: "Process-level proxy; summed RSS may double-count shared pages, assumes no unrelated Chrome process starts during the trial, and does not isolate GPU memory on unified-memory macOS",
            idle_rss_bytes: idle.bytes,
            peak_rss_bytes: peakBytes,
            incremental_peak_rss_bytes: peakBytes - idle.bytes,
            samples,
            in_page: inPage
        }
    } finally {
        if (interval) clearInterval(interval)
        await browser.close()
    }
}

const summarize = values => {
    const sorted = [...values].sort((left, right) => left - right)
    const quantile = probability => {
        const position = (sorted.length - 1) * probability
        const lower = Math.floor(position)
        const fraction = position - lower
        return sorted[lower + 1] === undefined
            ? sorted[lower]
            : sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower])
    }
    return {
        n: values.length,
        median_ms: quantile(0.5),
        q1_ms: quantile(0.25),
        q3_ms: quantile(0.75),
        p95_ms: quantile(0.95),
        mean_ms: values.reduce((sum, value) => sum + value, 0) / values.length,
        samples_ms: values
    }
}

const runColdTrials = async (backend, options, origin, example) => {
    const trials = []
    for (let trial = 0; trial < options.coldTrials; trial++) {
        console.log(`${backend}: cold trial ${trial + 1}/${options.coldTrials}`)
        const browser = await launchBrowser()
        try {
            const context = await browser.newContext()
            const page = await context.newPage()
            await loadHarness(page, origin, `${backend}:cold:${trial + 1}`)
            trials.push(await page.evaluate(
                configuration => window.runColdTrial(configuration),
                { backend, tokens: example.tokens, ages: example.ages_days }
            ))
            await context.close()
        } finally {
            await browser.close()
        }
    }
    return {
        context_length: example.tokens.length,
        trials,
        session_initialization: summarize(trials.map(trial => trial.initialize_ms)),
        first_inference: summarize(trials.map(trial => trial.first_inference_ms))
    }
}

const runFullEvaluation = async (backend, options, origin) => {
    console.log(`${backend}: starting full fidelity and latency evaluation`)
    const browser = await launchBrowser()
    try {
        const context = await browser.newContext()
        const page = await context.newPage()
        await loadHarness(page, origin, backend)
        const result = await page.evaluate(
            configuration => window.runEvaluation(configuration),
            {
                backend,
                manifestUrl: options.manifestUrl,
                warmups: options.warmups,
                runs: options.runs,
                trajectoryRuns: options.trajectoryRuns
            }
        )
        await context.close()
        return result
    } finally {
        await browser.close()
    }
}

const main = async () => {
    const options = parseArguments(process.argv.slice(2))
    const manifestPath = path.join(ROOT, options.manifestUrl.replace(/^\//, ""))
    const manifest = JSON.parse(await readFile(manifestPath, "utf8"))
    const provenance = await evaluationProvenance(manifest, manifestPath)
    const example = manifest.stratified.cases.find(caseRecord => caseRecord.context_length >= 24)
    if (!example) throw new Error("Manifest has no length-24 cold-start case")
    const { server, origin } = await startServer()
    await mkdir(path.join(ROOT, "evaluation", "results"), { recursive: true })
    try {
        for (const backend of options.backends) {
            const memoryExample = manifest.stratified.cases.find(
                caseRecord => caseRecord.context_length >= 48
            )
            if (!memoryExample) throw new Error("Manifest has no length-48 memory case")
            if (options.memoryOnly) {
                const processMemory = await runMemoryTrial(backend, origin, memoryExample)
                const memoryPath = path.join(
                    ROOT, "evaluation", "results", `browser_memory_${backend}.json`
                )
                await writeFile(memoryPath, `${JSON.stringify({
                    backend,
                    timestamp_utc: new Date().toISOString(),
                    evaluation_provenance: provenance,
                    process_memory: processMemory
                }, null, 2)}\n`)
                console.log(`${backend}: wrote ${memoryPath}`)
                continue
            }
            if (options.fidelityOnly) {
                console.log(`${backend}: refreshing fidelity evaluation`)
                const browser = await launchBrowser()
                let refreshed
                try {
                    const context = await browser.newContext()
                    const page = await context.newPage()
                    await loadHarness(page, origin, `${backend}:fidelity`)
                    refreshed = await page.evaluate(
                        configuration => window.runFidelityEvaluation(configuration),
                        { backend, manifestUrl: options.manifestUrl }
                    )
                    await context.close()
                } finally {
                    await browser.close()
                }
                const outputPath = path.join(ROOT, "evaluation", "results", `browser_${backend}.json`)
                const existing = JSON.parse(await readFile(outputPath, "utf8"))
                const output = {
                    ...existing,
                    fidelity: refreshed.fidelity,
                    preprocessing: refreshed.preprocessing,
                    determinism: refreshed.determinism,
                    environment: refreshed.environment,
                    evaluation_provenance: provenance,
                    fidelity_refreshed_utc: new Date().toISOString()
                }
                await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`)
                console.log(`${backend}: refreshed ${outputPath}`)
                continue
            }
            if (options.latencyOnly) {
                console.log(`${backend}: refreshing latency evaluation`)
                const browser = await launchBrowser()
                let latency
                try {
                    const context = await browser.newContext()
                    const page = await context.newPage()
                    await loadHarness(page, origin, `${backend}:latency`)
                    latency = await page.evaluate(
                        configuration => window.runLatencyEvaluation(configuration),
                        {
                            backend,
                            manifestUrl: options.manifestUrl,
                            warmups: options.warmups,
                            runs: options.runs,
                            trajectoryRuns: options.trajectoryRuns
                        }
                    )
                    await context.close()
                } finally {
                    await browser.close()
                }
                const outputPath = path.join(ROOT, "evaluation", "results", `browser_${backend}.json`)
                const existing = JSON.parse(await readFile(outputPath, "utf8"))
                const output = {
                    ...existing,
                    latency,
                    evaluation_provenance: provenance,
                    latency_refreshed_utc: new Date().toISOString()
                }
                await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`)
                console.log(`${backend}: refreshed ${outputPath}`)
                continue
            }
            const cold = await runColdTrials(backend, options, origin, example)
            const processMemory = await runMemoryTrial(backend, origin, memoryExample)
            const full = await runFullEvaluation(backend, options, origin)
            const output = {
                ...full,
                timestamp_utc: new Date().toISOString(),
                host: {
                    os_type: os.type(),
                    os_release: os.release(),
                    architecture: os.arch(),
                    logical_cpu_count: os.cpus().length,
                    cpu_model: os.cpus()[0]?.model,
                    total_memory_bytes: os.totalmem(),
                    power_condition: "AC power, battery charged at measurement start",
                    gpu_power_preference: "default (not explicitly set)"
                },
                evaluation_provenance: provenance,
                benchmark_configuration: {
                    cold_trials: options.coldTrials,
                    warmups: options.warmups,
                    runs_per_length: options.runs,
                    trajectory_runs: options.trajectoryRuns
                },
                cold_start: cold,
                process_memory: processMemory
            }
            const outputPath = path.join(ROOT, "evaluation", "results", `browser_${backend}.json`)
            await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`)
            const memoryPath = path.join(
                ROOT, "evaluation", "results", `browser_memory_${backend}.json`
            )
            await writeFile(memoryPath, `${JSON.stringify({
                backend,
                timestamp_utc: new Date().toISOString(),
                evaluation_provenance: provenance,
                process_memory: processMemory
            }, null, 2)}\n`)
            console.log(`${backend}: wrote ${outputPath}`)
        }
    } finally {
        await new Promise(resolve => server.close(resolve))
    }
}

main().catch(error => {
    console.error(error.stack || error)
    process.exitCode = 1
})
