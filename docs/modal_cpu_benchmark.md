# Modal CPU inference timing

The trained RealWaste YOLOv8n classifier was tested with one test image, Food Organics/Food Organics_157.jpg, using one first request followed by ten warm requests. Every request sent the image bytes to Modal and performed fresh inference. All predictions were Food Organics with confidence 0.995496, matching the image label. Repeated calls measure latency stability for this image.

## Measurements

- First request including container startup, imports and loading took 10.999 seconds.
- Warm end-to-end request median was 1.684 seconds.
- Warm request range was 1.567 to 2.350 seconds.
- Warm model.predict call median was 12.588 milliseconds, including preprocessing and postprocessing.
- Warm model forward inference median was 7.683 milliseconds.
- Dependency imports took 2.910 seconds at startup.
- Checkpoint loading took 2.227 seconds at startup.

## Configuration and scope

Modal CPU function requested one physical CPU core, 2 GiB RAM, and two PyTorch threads. All 11 calls reused one container. Remote dependencies were torch 2.15.0+cpu, Ultralytics 8.4.215, and Python 3.12.16. Input size was 224 pixels and inference device was CPU. The source image was decoded on the server for every request.

End-to-end timing includes the local Modal SDK request, image transfer, scheduling, remote processing, and response. One-time environment image builds and app creation were outside the timer. This run used SDK RPC. The subsequent [HTTP check](../results/metrics/demo_http_check.json) covered three models in a single request, and the [demo guide](../demo/README.md#validation) records Streamlit validation. Network conditions, input images, and container availability can change latency.

The initial run's prediction response failed to deserialize because torch.__version__ was returned as a PyTorch-specific object. The script now converts that metadata to plain text. The successful rerun provides all reported measurements.

The recorded CPU count reflects the host-visible count, while the requested allocation was one physical core. The app was run temporarily and stopped when the benchmark completed. GPU access was previously rejected because the account requires a payment method for GPU functions.

Machine-readable results are saved in results/metrics/modal_cpu_benchmark.json and results/metrics/modal_cpu_benchmark.csv. The checkpoint remains excluded from Git.
