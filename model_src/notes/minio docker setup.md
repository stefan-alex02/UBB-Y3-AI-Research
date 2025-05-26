You've landed on the **AIStor** page, which is MinIO's *enterprise, licensed* offering designed for very large-scale AI/ML workloads, often integrated with Kubernetes.

For your project, especially for development and even for many production use cases, you'll want to use the **standard, open-source MinIO server**, which is indeed **free to use**.

**Finding the Correct MinIO Download:**

1.  Go to the main MinIO website: [https://min.io/](https://min.io/)
2.  Look for a "Download" or "Get Started" section that points to the open-source server. It's usually very prominent.
    *   Direct link often looks like: [https://min.io/download](https://min.io/download) (but be careful you don't get redirected to a commercial product page).
    *   You should see options to download binaries for Windows, macOS, Linux, and Docker.

**Setup Method: OS (Windows) vs. Docker Container**

Both are viable, each with pros and cons:

*   **Directly on OS (Windows):**
    *   **Pros:**
        *   Simplest for absolute beginners if not familiar with Docker.
        *   Direct access to your local file system for data storage (e.g., `D:\minio_data`).
        *   No Docker overhead.
    *   **Cons:**
        *   Might pollute your global environment slightly (though MinIO is a single binary).
        *   Port conflicts can be more manual to resolve if other services use port 9000 or 9090.
        *   Less portable if you switch development machines or deploy to Linux later (though MinIO itself is cross-platform).

*   **Docker Container:**
    *   **Pros:**
        *   **Isolation:** Runs in a self-contained environment, not affecting your host OS directly.
        *   **Portability:** Easy to run the exact same setup on any machine with Docker (Windows, macOS, Linux).
        *   **Reproducibility:** Defined by a Docker command or `docker-compose.yml`.
        *   **Easy Updates/Version Management:** Docker makes it simple to pull new MinIO versions.
        *   **Clean Removal:** Just stop and remove the container.
    *   **Cons:**
        *   Requires Docker Desktop (or similar) to be installed and running.
        *   Slight learning curve if you're new to Docker (but it's a very valuable skill).
        *   Data persistence needs to be handled correctly using Docker volumes to ensure your MinIO data isn't lost when the container stops/restarts.

**Recommendation: Use Docker for MinIO Setup**

For long-term ease of use, portability, and isolation, **Docker is generally the recommended approach.** It's a standard tool in modern development.

**Setting up MinIO with Docker and Creating Buckets:**

1.  **Install Docker Desktop:** If you don't have it, download and install Docker Desktop for Windows from the official Docker website. Make sure it's running.

2.  **Create a Data Directory on Your Host:**
    Choose a place on your Windows machine where MinIO's data will be stored persistently. This is important so your data isn't lost when the Docker container stops.
    Example: `C:\minio_data` (or `D:\minio_data` etc.)
    *Create this folder if it doesn't exist.*

3.  **Run the MinIO Docker Container:**
    Open PowerShell or Command Prompt and run the following command:

    ```powershell
    docker run -p 9000:9000 -p 9091:9091 --name my-minio-server -e "MINIO_ROOT_USER=YOUR_MINIO_ACCESS_KEY" -e "MINIO_ROOT_PASSWORD=YOUR_MINIO_SECRET_KEY" -v C:\minio_data:/data quay.io/minio/minio server /data --console-address ":9091"
    ```

    Let's break this down:
    *   `docker run`: Command to run a new container.
    *   `-p 9000:9000`: Maps port 9000 on your host machine to port 9000 inside the container (for the S3 API).
    *   `-p 9091:9091`: Maps port 9091 on your host to port 9091 inside the container (for the MinIO Console/Web UI). *I'm using 9091 here instead of 9090 to avoid potential conflicts with other services that might default to 9090.*
    *   `--name my-minio-server`: Gives your container a friendly name.
    *   `-e "MINIO_ROOT_USER=YOUR_MINIO_ACCESS_KEY"`: Sets the root access key (username). **Replace `YOUR_MINIO_ACCESS_KEY`** with something like `minioadmin` or a strong random string.
    *   `-e "MINIO_ROOT_PASSWORD=YOUR_MINIO_SECRET_KEY"`: Sets the root secret key (password). **Replace `YOUR_MINIO_SECRET_KEY`** with a very strong password.
    *   `-v C:\minio_data:/data`: This is crucial for data persistence. It mounts the `C:\minio_data` directory from your host machine into the `/data` directory inside the container. MinIO will store its data in `/data`.
    *   `quay.io/minio/minio`: The official MinIO Docker image.
    *   `server /data`: Command to start the MinIO server and tell it to use the `/data` directory for storage.
    *   `--console-address ":9091"`: Tells MinIO to serve its web console on port 9091 inside the container.

4.  **Access MinIO Console:**
    Open your web browser and go to: `http://127.0.0.1:9091`
    Log in with the `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` you set in the Docker command.

5.  **Create Buckets:**
    Once logged into the MinIO Console:
    *   Click on the "Buckets" section in the left sidebar.
    *   Click the "+ Create Bucket" button.
    *   **Bucket 1: For User-Uploaded Images**
        *   Bucket Name: `user-images` (or choose a name from your schema, e.g., `cloud-app-user-images`)
        *   Versioning: Optional, can be useful.
        *   Object Locking: Probably not needed for this use case initially.
        *   Click "Create Bucket".
    *   **Bucket 2: For LIME Explanation Images (if storing them separately)**
        *   Bucket Name: `prediction-artifacts` (or `cloud-app-lime-images`)
        *   Click "Create Bucket".
    *   **Bucket 3: For ML Experiment Artifacts (Models, JSONs, Plots, Logs)**
        *   Bucket Name: `ml-experiment-artifacts` (or `cloud-app-ml-runs`)
        *   Click "Create Bucket".

    You now have your MinIO server running in Docker with the necessary buckets!

**Configuration in your Java Spring Boot Application (Example):**

Your `application.properties` or `application.yml` would look like this (assuming you used the keys/passwords from the Docker command):

```properties
# application.properties
aws.s3.endpoint=http://127.0.0.1:9000
aws.s3.access-key=YOUR_MINIO_ACCESS_KEY
aws.s3.secret-key=YOUR_MINIO_SECRET_KEY
aws.s3.region=us-east-1 # Or any valid AWS region string, MinIO doesn't enforce it
aws.s3.path-style-access=true # Important for MinIO

# You'll reference these bucket names in your service logic
app.s3.bucket.user-images=user-images
app.s3.bucket.prediction-artifacts=prediction-artifacts
app.s3.bucket.ml-experiment-artifacts=ml-experiment-artifacts
```

**Configuration in your Python Server (if it needs direct S3 access, e.g., to upload experiment artifacts):**

If your Python pipeline methods are modified to upload artifacts directly to S3 (instead of just returning S3 keys to Java which then does the upload), you'd use `boto3` in Python:

```python
import boto3

s3_client = boto3.client(
    's3',
    endpoint_url='http://127.0.0.1:9000',
    aws_access_key_id='YOUR_MINIO_ACCESS_KEY',
    aws_secret_access_key='YOUR_MINIO_SECRET_KEY',
    region_name='us-east-1' # Can be any valid region
)

# Example upload:
# s3_client.upload_file("local_file.json", "ml-experiment-artifacts", "dataset/model/exp_id/results.json")
```

**Is MinIO Free?**
Yes, the **standard MinIO server** (the one you download from `min.io/download` or use via `quay.io/minio/minio` Docker image) is **open-source (AGPLv3) and free to use**.
AIStor is their commercial, enterprise-focused product with additional features and support, which requires a license. For your project development and many production scenarios, the free, open-source MinIO server is perfectly adequate and very powerful.

This Docker setup provides a robust, isolated, and portable MinIO environment for your development.