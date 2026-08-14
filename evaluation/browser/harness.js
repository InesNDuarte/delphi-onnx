import DelphiONNX, { DEFAULT_IGNORE_TOKENS } from "/delphiSDK.js"

const VOCAB_SIZE = 1270
const HISTOGRAM_WIDTH = 1e-8
const HISTOGRAM_BINS = 100_002

const backendConfiguration = backend => {
    if (backend === "wasm") {
        return {
            executionProvider: "wasm",
            runtimeModuleURL: "/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs",
            wasmPaths: "/node_modules/onnxruntime-web/dist/",
            wasmNumThreads: 1,
            runtimeJavaScript: "/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs",
            runtimeBinary: "/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"
        }
    }
    if (backend === "webgpu") {
        return {
            executionProvider: "webgpu",
            runtimeModuleURL: "/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs",
            wasmPaths: "/node_modules/onnxruntime-web/dist/",
            wasmNumThreads: 1,
            runtimeJavaScript: "/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs",
            runtimeBinary: "/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm"
        }
    }
    throw new Error(`Unsupported backend: ${backend}`)
}

const createSdk = backend => {
    const configuration = backendConfiguration(backend)
    return new DelphiONNX({
        modelURL: "/delphi.onnx",
        seed: 20260813,
        ...configuration
    })
}

const fetchJson = async url => {
    const response = await fetch(url, { cache: "no-store" })
    if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`)
    return response.json()
}

const fetchFloat32 = async url => {
    const response = await fetch(url, { cache: "no-store" })
    if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`)
    const buffer = await response.arrayBuffer()
    if (buffer.byteLength % 4 !== 0) throw new Error(`${url} is not float32-aligned`)
    return new Float32Array(buffer)
}

const float32Bits = value => {
    const values = new Float32Array(1)
    values[0] = value
    return new Uint32Array(values.buffer)[0]
}

class ErrorMetrics {
    constructor() {
        this.count = 0
        this.absoluteSum = 0
        this.squareSum = 0
        this.maximum = 0
        this.withinTolerance = 0
        this.histogram = new Uint32Array(HISTOGRAM_BINS)
        this.maximumLocation = undefined
    }

    add(reference, candidate, location) {
        if (reference.length !== candidate.length) {
            throw new Error(`Shape mismatch at ${JSON.stringify(location)}: ${reference.length} != ${candidate.length}`)
        }
        for (let index = 0; index < reference.length; index++) {
            const expected = reference[index]
            const actual = candidate[index]
            const error = Math.abs(expected - actual)
            this.count++
            this.absoluteSum += error
            this.squareSum += error * error
            if (error <= 1e-4 + 1e-4 * Math.abs(expected)) this.withinTolerance++
            const bin = Math.min(Math.floor(error / HISTOGRAM_WIDTH), HISTOGRAM_BINS - 1)
            this.histogram[bin]++
            if (error > this.maximum) {
                this.maximum = error
                this.maximumLocation = {
                    ...location,
                    flat_index: index,
                    position: location.single_position ? location.position : Math.floor(index / VOCAB_SIZE),
                    token: index % VOCAB_SIZE,
                    reference: expected,
                    candidate: actual
                }
            }
        }
    }

    quantile(probability) {
        const target = Math.ceil(this.count * probability)
        let cumulative = 0
        for (let bin = 0; bin < this.histogram.length; bin++) {
            cumulative += this.histogram[bin]
            if (cumulative >= target) return (bin + 0.5) * HISTOGRAM_WIDTH
        }
        return Number.NaN
    }

    summary() {
        return {
            count: this.count,
            mean_absolute_error: this.absoluteSum / this.count,
            root_mean_square_error: Math.sqrt(this.squareSum / this.count),
            median_absolute_error_approx: this.quantile(0.5),
            p95_absolute_error_approx: this.quantile(0.95),
            p99_absolute_error_approx: this.quantile(0.99),
            histogram_resolution: HISTOGRAM_WIDTH,
            maximum_absolute_error: this.maximum,
            maximum_error_location: this.maximumLocation,
            atol: 1e-4,
            rtol: 1e-4,
            within_tolerance_fraction: this.withinTolerance / this.count
        }
    }
}

const quantile = (values, probability) => {
    if (!values.length) return Number.NaN
    const copy = [...values].sort((left, right) => left - right)
    const position = (copy.length - 1) * probability
    const lower = Math.floor(position)
    const fraction = position - lower
    return copy[lower + 1] === undefined
        ? copy[lower]
        : copy[lower] + fraction * (copy[lower + 1] - copy[lower])
}

const sampleSummary = (values, includeSamples = true) => {
    const summary = {
        n: values.length,
        mean: values.reduce((sum, value) => sum + value, 0) / values.length,
        median: quantile(values, 0.5),
        q1: quantile(values, 0.25),
        q3: quantile(values, 0.75),
        p95: quantile(values, 0.95),
        maximum: Math.max(...values)
    }
    if (includeSamples) summary.samples = values
    return summary
}

const expectedMask = (tokens, ignoreTokens) => {
    const mask = new Uint8Array(VOCAB_SIZE)
    for (const token of ignoreTokens) mask[token] = 1
    for (const token of tokens) if (token > 1 && token < VOCAB_SIZE) mask[token] = 1
    return mask
}

const stableDistribution = (logits, mask) => {
    let maximum = -Infinity
    for (let index = 0; index < logits.length; index++) {
        if (!mask[index] && logits[index] > maximum) maximum = logits[index]
    }
    const probabilities = new Float64Array(logits.length)
    let total = 0
    for (let index = 0; index < logits.length; index++) {
        if (!mask[index]) {
            const value = Math.exp(logits[index] - maximum)
            probabilities[index] = value
            total += value
        }
    }
    for (let index = 0; index < probabilities.length; index++) probabilities[index] /= total
    return { probabilities, logTotalRate: maximum + Math.log(total) }
}

const topK = (values, k) => {
    const bestIndices = new Int32Array(k)
    const bestValues = new Float64Array(k)
    bestIndices.fill(-1)
    bestValues.fill(-Infinity)
    for (let index = 0; index < values.length; index++) {
        const value = values[index]
        if (value <= bestValues[k - 1]) continue
        let position = k - 1
        while (position > 0 && value > bestValues[position - 1]) {
            bestValues[position] = bestValues[position - 1]
            bestIndices[position] = bestIndices[position - 1]
            position--
        }
        bestValues[position] = value
        bestIndices[position] = index
    }
    return [...bestIndices]
}

class DistributionMetrics {
    constructor() {
        this.contexts = 0
        this.probabilityCount = 0
        this.probabilityAbsoluteSum = 0
        this.probabilityMaximum = 0
        this.totalVariation = []
        this.logTotalRateError = []
        this.jensenShannon = []
        this.top1Matches = 0
        this.top5Overlap = []
        this.top10Overlap = []
        this.maskMismatches = 0
        this.tvFailures = 0
        this.probabilityFailures = 0
        this.logRateFailures = 0
    }

    add(reference, candidate, tokens, sdk, ignoreTokens) {
        const mask = expectedMask(tokens, ignoreTokens)
        const sdkMasked = sdk.applyPreSamplingMask(candidate, tokens, {
            ignoreTokens,
            noRepeat: true
        })
        for (let index = 0; index < mask.length; index++) {
            if ((sdkMasked[index] === -Infinity) !== Boolean(mask[index])) this.maskMismatches++
        }
        const left = stableDistribution(reference, mask)
        const right = stableDistribution(candidate, mask)
        let absoluteSum = 0
        let maximum = 0
        let js = 0
        for (let index = 0; index < VOCAB_SIZE; index++) {
            const p = left.probabilities[index]
            const q = right.probabilities[index]
            const error = Math.abs(p - q)
            absoluteSum += error
            maximum = Math.max(maximum, error)
            const midpoint = 0.5 * (p + q)
            if (p > 0) js += 0.5 * p * Math.log(p / midpoint)
            if (q > 0) js += 0.5 * q * Math.log(q / midpoint)
        }
        const tv = 0.5 * absoluteSum
        const logError = Math.abs(left.logTotalRate - right.logTotalRate)
        this.contexts++
        this.probabilityCount += VOCAB_SIZE
        this.probabilityAbsoluteSum += absoluteSum
        this.probabilityMaximum = Math.max(this.probabilityMaximum, maximum)
        this.totalVariation.push(tv)
        this.logTotalRateError.push(logError)
        this.jensenShannon.push(js)
        this.tvFailures += Number(tv > 1e-4)
        this.probabilityFailures += Number(maximum > 1e-4)
        this.logRateFailures += Number(logError > 1e-4)
        const left10 = topK(left.probabilities, 10)
        const right10 = topK(right.probabilities, 10)
        this.top1Matches += Number(left10[0] === right10[0])
        this.top5Overlap.push(left10.slice(0, 5).filter(value => right10.slice(0, 5).includes(value)).length / 5)
        this.top10Overlap.push(left10.filter(value => right10.includes(value)).length / 10)
    }

    summary() {
        return {
            contexts: this.contexts,
            mask_cell_mismatches: this.maskMismatches,
            mean_absolute_probability_error: this.probabilityAbsoluteSum / this.probabilityCount,
            maximum_absolute_probability_error: this.probabilityMaximum,
            total_variation_distance: sampleSummary(this.totalVariation, false),
            jensen_shannon_divergence: sampleSummary(this.jensenShannon, false),
            absolute_log_total_rate_error: sampleSummary(this.logTotalRateError, false),
            top1_agreement: this.top1Matches / this.contexts,
            mean_top5_overlap: this.top5Overlap.reduce((sum, value) => sum + value, 0) / this.contexts,
            mean_top10_overlap: this.top10Overlap.reduce((sum, value) => sum + value, 0) / this.contexts,
            acceptance_failures: {
                total_variation_over_1e_4: this.tvFailures,
                event_probability_error_over_1e_4: this.probabilityFailures,
                log_total_rate_error_over_1e_4: this.logRateFailures
            }
        }
    }
}

const memorySnapshot = async label => {
    if (typeof performance.measureUserAgentSpecificMemory !== "function") {
        return { label, available: false }
    }
    try {
        const result = await performance.measureUserAgentSpecificMemory()
        return { label, available: true, bytes: result.bytes }
    } catch (error) {
        return { label, available: false, error: String(error) }
    }
}

const sizeOf = async url => {
    const response = await fetch(url, { cache: "no-store" })
    if (!response.ok) throw new Error(`Failed to load artifact ${url}`)
    return (await response.arrayBuffer()).byteLength
}

const artifactSizes = async backend => {
    const configuration = backendConfiguration(backend)
    const paths = {
        model: "/delphi.onnx",
        sdk: "/delphiSDK.js",
        labels: "/delphi_labels_chapters_colours_icd.json",
        runtime_javascript: configuration.runtimeJavaScript,
        runtime_wasm: configuration.runtimeBinary
    }
    const entries = await Promise.all(Object.entries(paths).map(async ([name, path]) => [name, await sizeOf(path)]))
    const bytes = Object.fromEntries(entries)
    bytes.first_use_payload = Object.values(bytes).reduce((sum, value) => sum + value, 0)
    return bytes
}

const summarizeMilliseconds = values => {
    const raw = sampleSummary(values)
    return {
        n: raw.n,
        median_ms: raw.median,
        q1_ms: raw.q1,
        q3_ms: raw.q3,
        p95_ms: raw.p95,
        mean_ms: raw.mean,
        samples_ms: raw.samples
    }
}

const timeAsync = async functionToTime => {
    const start = performance.now()
    await functionToTime()
    return performance.now() - start
}

const contextForLength = (caseRecord, length) => ({
    tokens: caseRecord.tokens.slice(-length),
    ages: caseRecord.ages_days.slice(-length)
})

const benchmark = async (sdk, manifest, options, firstInferenceMs) => {
    const casesById = new Map(manifest.stratified.cases.map(caseRecord => [caseRecord.id, caseRecord]))
    const warmupRuns = options.warmups
    const measuredRuns = options.runs
    const results = {
        definitions: {
            model_only: "Prebuilt ORT tensors through session.run and CPU output materialization",
            end_to_end_step: "SDK arrays through tensor construction, session.run, readback, and final-position extraction",
            paired_order: "Model-only and end-to-end calls were paired with alternating order",
            full_trajectory: "Actual SDK generation from 16 events for exactly 32 sampled events plus final all-position inference (33 graph executions)"
        },
        first_inference_ms: firstInferenceMs,
        warmup_runs: warmupRuns,
        measured_runs_per_length: measuredRuns,
        model_only: {},
        end_to_end_step: {}
    }
    let totalWarmup = 0
    for (const length of [12, 24, 48]) {
        const ids = manifest.benchmark_context_ids[String(length)]
        const contexts = ids.map(id => contextForLength(casesById.get(id), length))
        const feeds = contexts.map(context => sdk.createInputTensors(context.tokens, context.ages))
        for (let run = 0; run < warmupRuns; run++) {
            totalWarmup += await timeAsync(() => sdk.runInputTensors(feeds[run % feeds.length]))
            totalWarmup += await timeAsync(() => {
                const context = contexts[run % contexts.length]
                return sdk.getNextLogits(context.tokens, context.ages)
            })
        }
        const modelSamples = []
        const endToEndSamples = []
        for (let run = 0; run < measuredRuns; run++) {
            const context = contexts[run % contexts.length]
            const runModel = () => sdk.runInputTensors(feeds[run % feeds.length])
            const runEndToEnd = () => sdk.getNextLogits(context.tokens, context.ages)
            // Alternate paired measurement order to limit thermal/order bias.
            if (run % 2 === 0) {
                modelSamples.push(await timeAsync(runModel))
                endToEndSamples.push(await timeAsync(runEndToEnd))
            } else {
                endToEndSamples.push(await timeAsync(runEndToEnd))
                modelSamples.push(await timeAsync(runModel))
            }
        }
        results.model_only[String(length)] = summarizeMilliseconds(modelSamples)
        results.end_to_end_step[String(length)] = summarizeMilliseconds(endToEndSamples)
    }
    results.warmup_total_ms = totalWarmup

    const rolloutCase = manifest.stratified.cases.find(caseRecord => caseRecord.context_length >= 48)
    const prefix = contextForLength(rolloutCase, 48)
    prefix.tokens = prefix.tokens.slice(0, 16)
    prefix.ages = prefix.ages.slice(0, 16)
    const trajectorySamples = []
    for (let warmup = 0; warmup < Math.min(2, warmupRuns); warmup++) {
        await sdk.generateTrajectory(prefix.tokens, prefix.ages, {
            maxNewTokens: 32,
            maxAge: Number.MAX_VALUE,
            terminationTokens: [-1]
        })
    }
    for (let run = 0; run < options.trajectoryRuns; run++) {
        trajectorySamples.push(await timeAsync(() => sdk.generateTrajectory(
            prefix.tokens,
            prefix.ages,
            { maxNewTokens: 32, maxAge: Number.MAX_VALUE, terminationTokens: [-1] }
        )))
    }
    results.full_trajectory = summarizeMilliseconds(trajectorySamples)
    results.full_trajectory.generated_events = 32
    results.full_trajectory.graph_executions = 33
    return results
}

const determinismCheck = async (sdk, caseRecord) => {
    const context = contextForLength(caseRecord, Math.min(24, caseRecord.context_length))
    const baseline = (await sdk.getAllLogits(context.tokens, context.ages)).logits
    let bitwiseEqualRuns = 0
    let maximumAbsoluteDifference = 0
    for (let run = 0; run < 10; run++) {
        const candidate = (await sdk.getAllLogits(context.tokens, context.ages)).logits
        let equal = candidate.length === baseline.length
        for (let index = 0; index < baseline.length; index++) {
            const difference = Math.abs(candidate[index] - baseline[index])
            maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, difference)
            if (float32Bits(candidate[index]) !== float32Bits(baseline[index])) equal = false
        }
        bitwiseEqualRuns += Number(equal)
    }
    return { repeated_runs: 10, bitwise_equal_runs: bitwiseEqualRuns, maximum_absolute_difference: maximumAbsoluteDifference }
}

const preprocessingCheck = (sdk, cases) => {
    const result = {
        cases: cases.length,
        token_cells: 0,
        token_mismatches: 0,
        age_cells: 0,
        float32_age_bit_mismatches: 0,
        postprocessed_event_name_mismatches: 0
    }
    for (const caseRecord of cases) {
        const eventsList = caseRecord.events.map((event, index) => ({ event, age: caseRecord.ages_years[index] }))
        const prepared = sdk.prepareTrajectoryInputs(eventsList, { ageUnit: "years" })
        result.token_cells += caseRecord.tokens.length
        result.age_cells += caseRecord.ages_days.length
        for (let index = 0; index < caseRecord.tokens.length; index++) {
            result.token_mismatches += Number(prepared.eventTokens[index] !== caseRecord.tokens[index])
            result.float32_age_bit_mismatches += Number(
                float32Bits(prepared.ages[index]) !== float32Bits(caseRecord.ages_days[index])
            )
        }
        const names = sdk.getEventsFromTokens(prepared.eventTokens)
        for (let index = 0; index < names.length; index++) {
            result.postprocessed_event_name_mismatches += Number(names[index] !== caseRecord.events[index])
        }
    }
    result.exact = result.token_mismatches === 0 &&
        result.float32_age_bit_mismatches === 0 &&
        result.postprocessed_event_name_mismatches === 0
    return result
}

const evaluateFidelity = async (sdk, manifest, manifestUrl) => {
    const base = new URL(manifestUrl, location.href)
    const allReferenceUrl = new URL(manifest.all_cohort.reference_file, base)
    const fullReferenceUrl = new URL(manifest.stratified.cases[0].logits_file, base)
    console.log("Loading PyTorch reference logits")
    const [allReference, fullReference] = await Promise.all([
        fetchFloat32(allReferenceUrl),
        fetchFloat32(fullReferenceUrl)
    ])
    const ignoreTokens = manifest.checkpoint.model_args.ignore_tokens
    const finalErrors = new ErrorMetrics()
    const finalDistributions = new DistributionMetrics()
    const fullErrors = new ErrorMetrics()
    const fullDistributions = new DistributionMetrics()

    console.log(`Evaluating ${manifest.all_cohort.n_cases} final contexts`)
    for (let row = 0; row < manifest.all_cohort.cases.length; row++) {
        const caseRecord = manifest.all_cohort.cases[row]
        const output = await sdk.getNextLogits(caseRecord.tokens, caseRecord.ages_days)
        const candidate = output.logits
        const offset = caseRecord.reference_row * VOCAB_SIZE
        const reference = allReference.subarray(offset, offset + VOCAB_SIZE)
        finalErrors.add(reference, candidate, {
            case_id: String(caseRecord.patient_id),
            position: caseRecord.context_length - 1,
            single_position: true
        })
        finalDistributions.add(reference, candidate, caseRecord.tokens, sdk, ignoreTokens)
        if ((row + 1) % 500 === 0) console.log(`Final fidelity ${row + 1}/${manifest.all_cohort.n_cases}`)
    }

    console.log(`Evaluating ${manifest.stratified.cases.length} full-position contexts`)
    for (let caseIndex = 0; caseIndex < manifest.stratified.cases.length; caseIndex++) {
        const caseRecord = manifest.stratified.cases[caseIndex]
        const output = await sdk.getAllLogits(caseRecord.tokens, caseRecord.ages_days)
        const reference = fullReference.subarray(
            caseRecord.logits_element_offset,
            caseRecord.logits_element_offset + caseRecord.logits_element_count
        )
        fullErrors.add(reference, output.logits, { case_id: caseRecord.id, single_position: false })
        for (let position = 0; position < caseRecord.context_length; position++) {
            const start = position * VOCAB_SIZE
            fullDistributions.add(
                reference.subarray(start, start + VOCAB_SIZE),
                output.logits.subarray(start, start + VOCAB_SIZE),
                caseRecord.tokens.slice(0, position + 1),
                sdk,
                ignoreTokens
            )
        }
        if ((caseIndex + 1) % 32 === 0) console.log(`Full fidelity ${caseIndex + 1}/${manifest.stratified.cases.length}`)
    }
    return {
        all_cohort_final_logits: finalErrors.summary(),
        all_cohort_final_distributions: finalDistributions.summary(),
        stratified_all_position_logits: fullErrors.summary(),
        stratified_all_position_distributions: fullDistributions.summary()
    }
}

const webGpuMetadata = sdk => {
    if (sdk.executionProvider !== "webgpu") return undefined
    const adapter = sdk.runtime?.env?.webgpu?.adapter
    const info = adapter?.info
    if (!info) {
        return {
            session_created_with_webgpu_provider: true,
            adapter_metadata_exposed_by_runtime: false
        }
    }
    return {
        session_created_with_webgpu_provider: true,
        adapter_metadata_exposed_by_runtime: true,
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description
    }
}

window.runColdTrial = async ({ backend, tokens, ages }) => {
    const baselineMemory = await memorySnapshot("baseline")
    const sdk = createSdk(backend)
    const initializeMs = await timeAsync(() => sdk.initialize())
    const afterInitializeMemory = await memorySnapshot("after_initialize")
    const firstInferenceMs = await timeAsync(() => sdk.getNextLogits(tokens, ages))
    return {
        initialize_ms: initializeMs,
        first_inference_ms: firstInferenceMs,
        memory: [baselineMemory, afterInitializeMemory],
        webgpu: webGpuMetadata(sdk)
    }
}

window.runMemoryWorkload = async ({ backend, tokens, ages }) => {
    const snapshots = [await memorySnapshot("idle_page")]
    const sdk = createSdk(backend)
    await sdk.initialize()
    snapshots.push(await memorySnapshot("after_session_initialization"))
    for (let run = 0; run < 10; run++) await sdk.getNextLogits(tokens, ages)
    snapshots.push(await memorySnapshot("after_ten_warm_inferences"))
    await sdk.generateTrajectory(tokens.slice(0, 16), ages.slice(0, 16), {
        maxNewTokens: 32,
        maxAge: Number.MAX_VALUE,
        terminationTokens: [-1]
    })
    snapshots.push(await memorySnapshot("after_32_event_rollout"))
    return { user_agent_specific_memory: snapshots, webgpu: webGpuMetadata(sdk) }
}

window.runFidelityEvaluation = async options => {
    const manifestUrl = options.manifestUrl || "/evaluation/generated/manifest.json"
    const manifest = await fetchJson(manifestUrl)
    const sdk = createSdk(options.backend)
    await sdk.initialize()
    const firstCase = manifest.stratified.cases.find(caseRecord => caseRecord.context_length >= 24)
    return {
        fidelity: await evaluateFidelity(sdk, manifest, manifestUrl),
        preprocessing: preprocessingCheck(sdk, manifest.stratified.cases),
        determinism: await determinismCheck(sdk, firstCase),
        environment: {
            user_agent: navigator.userAgent,
            platform: navigator.platform,
            hardware_concurrency: navigator.hardwareConcurrency,
            cross_origin_isolated: crossOriginIsolated,
            secure_context: isSecureContext,
            navigator_gpu: Boolean(navigator.gpu),
            onnxruntime_web: "1.27.0",
            wasm_num_threads: options.backend === "wasm" ? 1 : null,
            webgpu: webGpuMetadata(sdk)
        }
    }
}

window.runLatencyEvaluation = async options => {
    const manifestUrl = options.manifestUrl || "/evaluation/generated/manifest.json"
    const manifest = await fetchJson(manifestUrl)
    const sdk = createSdk(options.backend)
    await sdk.initialize()
    const firstCase = manifest.stratified.cases.find(caseRecord => caseRecord.context_length >= 24)
    const firstContext = contextForLength(firstCase, 24)
    const firstInferenceMs = await timeAsync(
        () => sdk.getNextLogits(firstContext.tokens, firstContext.ages)
    )
    return benchmark(sdk, manifest, options, firstInferenceMs)
}

window.runEvaluation = async options => {
    const manifestUrl = options.manifestUrl || "/evaluation/generated/manifest.json"
    const manifest = await fetchJson(manifestUrl)
    const baselineMemory = await memorySnapshot("baseline")
    const sdk = createSdk(options.backend)
    const initializeMs = await timeAsync(() => sdk.initialize())
    const afterInitializeMemory = await memorySnapshot("after_initialize")
    const firstCase = manifest.stratified.cases.find(caseRecord => caseRecord.context_length >= 24)
    const firstContext = contextForLength(firstCase, 24)
    const firstInferenceMs = await timeAsync(() => sdk.getNextLogits(firstContext.tokens, firstContext.ages))
    console.log(`${options.backend}: initialized; first inference ${firstInferenceMs.toFixed(3)} ms`)
    const latency = await benchmark(sdk, manifest, options, firstInferenceMs)
    const afterBenchmarkMemory = await memorySnapshot("after_benchmark")
    const determinism = await determinismCheck(sdk, firstCase)
    const preprocessing = preprocessingCheck(sdk, manifest.stratified.cases)
    const fidelity = await evaluateFidelity(sdk, manifest, manifestUrl)
    const afterFidelityMemory = await memorySnapshot("after_fidelity")
    return {
        schema_version: 1,
        backend: options.backend,
        environment: {
            user_agent: navigator.userAgent,
            platform: navigator.platform,
            hardware_concurrency: navigator.hardwareConcurrency,
            cross_origin_isolated: crossOriginIsolated,
            secure_context: isSecureContext,
            navigator_gpu: Boolean(navigator.gpu),
            onnxruntime_web: "1.27.0",
            wasm_num_threads: options.backend === "wasm" ? 1 : null,
            webgpu: webGpuMetadata(sdk)
        },
        artifacts_bytes: await artifactSizes(options.backend),
        session_initialization_ms: initializeMs,
        latency,
        memory: [baselineMemory, afterInitializeMemory, afterBenchmarkMemory, afterFidelityMemory],
        determinism,
        preprocessing,
        fidelity
    }
}

window.evaluationHarnessReady = true
