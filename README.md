# vlm-indoor-localization

This repository is the paper-release version of our indoor localization pipeline for blind and low-vision navigation support. It contains one reproducible end-to-end setup:

- `gpt-5-mini` for image descriptions
- `gpt-5` for final localization reasoning
- Pinecone retrieval over node visual features
- `softmax_aggregate` heuristic weighting for combining retrieval evidence

The repository is organized so a user can start from raw images, generate descriptions, build the index, run localization, and evaluate predictions by adding API keys to a `.env` file.

## Repository Layout

- `data/`: floor-specific images, coordinates, floorplans, and description files
- `prompts/`: prompt used for image description generation
- `configs/paper_release.yaml`: the single paper-release configuration
- `RAG/`: reusable Python modules
- `scripts/`: command-line entry points
- `outputs/`: generated localization predictions and evaluation outputs

## Dataset Download

The dataset is not stored in Git. Download it separately and place the extracted floor folders under `data/`.

Google Drive:

`https://drive.google.com/drive/folders/1nD8qnsbukBmhBmdfeqi9hmT5QiMFA5i4?usp=sharing`

After downloading, your local repository should look like:

```text
data/
├── Lighthouse - Floor 3/
├── Lighthouse - Floor 6/
├── CEPSR - Floor 7/
├── ACC - Floor 17/
└── 6 Metrotech - Floor 4/
```

## Requirements

- Python 3.10+
- An OpenAI API key
- A Pinecone API key

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the repository root:

```bash
cp .env.example .env
```

Then fill in:

```env
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

## Paper Pipeline

The default paper setting is defined in [`configs/paper_release.yaml`](/Users/shachafhaviv/Projects/vlm-indoor-localization/configs/paper_release.yaml):

- description model: `gpt-5-mini`
- reasoning model: `gpt-5`
- embedding model: `all-MiniLM-L6-v2`
- retrieval context size: `topk = 10`
- aggregation: `softmax_aggregate`
- runs per query: `2`

## End-to-End Usage

### 1. Generate node descriptions

Example for Lighthouse Floor 3:

```bash
python3 scripts/generate_descriptions.py \
  --config configs/paper_release.yaml \
  --floor "Lighthouse - Floor 3" \
  --image-kind node
```

### 2. Generate query descriptions

Standard query set:

```bash
python3 scripts/generate_descriptions.py \
  --config configs/paper_release.yaml \
  --floor "Lighthouse - Floor 3" \
  --image-kind query \
  --query-set standard
```

Extra query set:

```bash
python3 scripts/generate_descriptions.py \
  --config configs/paper_release.yaml \
  --floor "Lighthouse - Floor 3" \
  --image-kind query \
  --query-set extra
```

### 3. Build the Pinecone index

```bash
python3 scripts/build_index.py \
  --config configs/paper_release.yaml \
  --floor "Lighthouse - Floor 3"
```

### 4. Run localization

```bash
python3 scripts/run_localization.py \
  --config configs/paper_release.yaml \
  --floor "Lighthouse - Floor 3" \
  --query-set standard
```

This writes predictions to `outputs/`.

Optional: include distance estimation against the predicted node image:

```bash
python3 scripts/run_localization.py \
  --config configs/paper_release.yaml \
  --floor "Lighthouse - Floor 3" \
  --query-set standard \
  --include-distance
```

When `--include-distance` is enabled, the output CSV also includes:

- `predicted_distance_m`
- `distance_reason`
- `distance_relative_location`
- `distance_confidence`
- `distance_key_cues`

### 5. Evaluate localization predictions

```bash
python3 scripts/evaluate_localization.py \
  --predictions outputs/lighthouse-floor-3_gpt-5_top10_standard.csv
```

This writes:

- a detailed per-query CSV
- a summary JSON with location, direction, and total accuracy

Optional: also evaluate the distance stage:

```bash
python3 scripts/evaluate_localization.py \
  --predictions outputs/lighthouse-floor-3_gpt-5_top10_standard.csv \
  --config configs/paper_release.yaml \
  --floor "Lighthouse - Floor 3" \
  --query-set standard \
  --evaluate-distance
```

Distance evaluation is optional and only runs when all of the following are available:

- the predictions CSV contains distance predictions from `--include-distance`
- the floor config includes `meters_per_pixel`
- the floor has coordinate CSVs inside its configured `images_locations_dir`

If these inputs are missing, the main localization evaluation still runs and the summary JSON records that distance evaluation was skipped.

## Supported Floors

The release config currently includes:

- `Lighthouse - Floor 3`
- `Lighthouse - Floor 6`
- `CEPSR - Floor 7`
- `ACC - Floor 17`
- `6 Metrotech - Floor 4`

## Adding A New Floor

To run the pipeline on a floor that is not already listed in the config, you need to do two things:

1. Create a new floor data folder under `data/`
2. Add a matching entry under `floors:` in [`configs/paper_release.yaml`](/Users/shachafhaviv/Projects/vlm-indoor-localization/configs/paper_release.yaml)

### Required folder structure

Each new floor should follow the same layout as the existing floors. A typical layout looks like:

```text
data/<New Floor>/
├── Node images/
├── Query images/
├── Query images old/              # or another folder used for the extra query set
├── descriptions/
│   ├── node_images_gpt_5_mini.json
│   ├── query_images_gpt_5_mini.json
│   ├── query_extra_images_gpt_5_mini.json
│   └── node_relative_locations.txt
├── Images locations/              # folder name must match your config entry
│   ├── nodes_locations_coordinates.csv
│   ├── query_locations_coordinates.csv
│   └── query_extra_locations_coordinates.csv
└── Floorplan/
```

Notes:

- The exact folder names are configurable, but they must match the paths in [`configs/paper_release.yaml`](/Users/shachafhaviv/Projects/vlm-indoor-localization/configs/paper_release.yaml).
- The description JSON files do not need to exist before running the description-generation scripts; they will be created automatically.
- The `node_relative_locations.txt` file must be created by you for the new floor.

### Filename conventions

The scripts expect image filenames to encode ground-truth metadata.

Node images must use this format:

```text
node<NUMBER>_<north|south|east|west>.<jpg|jpeg|png>
```

Example:

```text
node12_north.jpg
```

Query image filenames must also contain:

- the true node number or numbers
- the facing direction as one of `north`, `south`, `east`, or `west`

Examples:

```text
node9_south.jpg
node15_16_south.jpg
node3_west.png
```

This matters because evaluation reads the true location and direction directly from the query image filename.

### Add the floor to the config

Add a new entry under `floors:` in [`configs/paper_release.yaml`](/Users/shachafhaviv/Projects/vlm-indoor-localization/configs/paper_release.yaml). For example:

```yaml
floors:
  "My Building - Floor 1":
    path: data/My Building - Floor 1
    index_name: my-building-floor-1
    meters_per_pixel: 0.02
    node_descriptions: descriptions/node_images_gpt_5_mini.json
    query_files:
      standard: descriptions/query_images_gpt_5_mini.json
      extra: descriptions/query_extra_images_gpt_5_mini.json
    relative_locations_file: descriptions/node_relative_locations.txt
    images_locations_dir: data/My Building - Floor 1/Images locations
    node_images_dir: Node images
    query_images_dir:
      standard: Query images
      extra: Query images old
```

Field meanings:

- `path`: root folder for the floor data
- `index_name`: Pinecone index name for this floor
- `meters_per_pixel`: required only if you want distance evaluation
- `node_descriptions`: output JSON for generated node descriptions
- `query_files`: output JSONs for generated standard and extra query descriptions
- `relative_locations_file`: text file describing relative relationships between nodes
- `images_locations_dir`: folder containing coordinate CSVs
- `node_images_dir`: folder containing node images
- `query_images_dir`: folders containing standard and extra query images

### Minimum setup checklist

Before running the pipeline on a new floor, make sure you have:

- node images in the configured `node_images_dir`
- query images in the configured standard query directory
- optional extra query images in the configured extra query directory
- a `node_relative_locations.txt` file
- a matching floor entry in [`configs/paper_release.yaml`](/Users/shachafhaviv/Projects/vlm-indoor-localization/configs/paper_release.yaml)
- valid filename conventions for both node and query images

If you also want distance evaluation for a new floor, make sure you have:

- `meters_per_pixel` set for that floor in [`configs/paper_release.yaml`](/Users/shachafhaviv/Projects/vlm-indoor-localization/configs/paper_release.yaml)
- `nodes_locations_coordinates.csv` inside the configured images-locations folder
- `query_locations_coordinates.csv` for the standard query set
- `query_extra_locations_coordinates.csv` for the extra query set if you evaluate extra queries

### Running the pipeline for a new floor

Once the new floor has been added to the config, the workflow is the same as for the existing floors:

1. Generate node descriptions
2. Generate query descriptions
3. Build the Pinecone index
4. Run localization
5. Evaluate predictions

Example:

```bash
python3 scripts/generate_descriptions.py \
  --config configs/paper_release.yaml \
  --floor "My Building - Floor 1" \
  --image-kind node

python3 scripts/generate_descriptions.py \
  --config configs/paper_release.yaml \
  --floor "My Building - Floor 1" \
  --image-kind query \
  --query-set standard

python3 scripts/build_index.py \
  --config configs/paper_release.yaml \
  --floor "My Building - Floor 1"

python3 scripts/run_localization.py \
  --config configs/paper_release.yaml \
  --floor "My Building - Floor 1" \
  --query-set standard
```

## Notes On Reproducibility

- Node retrieval is filtered to descriptions generated by `gpt_5_mini`.
- Query evaluation uses the ground-truth location and direction encoded in each query filename.
- The repository keeps only the best pipeline from the paper, rather than the full experimental matrix.
- Existing description JSON files are included, but the scripts also let users regenerate them from the raw images.

## Output Format

Localization output CSVs contain:

- query image name
- run number
- predicted location
- predicted direction
- reasoning text
- raw model answer
- top retrieved candidates

Evaluation output includes:

- per-query correctness flags
- location accuracy
- direction accuracy conditioned on correct location
- total accuracy
