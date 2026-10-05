import requests
import os
import time

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

SAMPLES_PER_GENRE = 10

BASE_URL = "https://datasets-server.huggingface.co/rows"

counts = {genre: 0 for genre in GENRES}

print("Starting download...")
print("Downloading only 10 clips per genre.\n")

# Check dataset in batches of 100 rows
for offset in range(0, 3000, 100):

    if all(counts[g] >= SAMPLES_PER_GENRE for g in GENRES):
        break

    print(f"Checking rows {offset} - {offset + 99}...")

    url = (
        f"{BASE_URL}"
        f"?dataset=Feanix%2Fgtzan-10-sec"
        f"&config=default"
        f"&split=train"
        f"&offset={offset}"
        f"&length=100"
    )

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

    except Exception as e:
        print("API error:", e)
        continue

    rows = data.get("rows", [])

    for item in rows:

        row = item["row"]

        # Genre can be numeric ClassLabel
        genre_value = row.get("genre")

        if isinstance(genre_value, int):
            genre = GENRES[genre_value]
        else:
            genre = str(genre_value)

        if genre not in GENRES:
            continue

        if counts[genre] >= SAMPLES_PER_GENRE:
            continue

        # Get audio information
        audio = row.get("audio")

        # Hugging Face may return audio as a list
        if isinstance(audio, list):
            if len(audio) == 0:
                continue
            audio = audio[0]

        if isinstance(audio, dict):
            audio_url = audio.get("src")

            # Some responses may provide a URL differently
            if not audio_url:
                audio_url = audio.get("url")

        else:
            audio_url = None

        if not audio_url:
            print(f"Audio URL not found for {genre}")
            continue

        folder = os.path.join("dataset", genre)
        os.makedirs(folder, exist_ok=True)

        filename = f"{genre}_{counts[genre]:03d}.wav"
        output_path = os.path.join(folder, filename)

        try:
            print(f"  Downloading {filename}...")

            audio_response = requests.get(
                audio_url,
                timeout=120
            )

            audio_response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(audio_response.content)

            counts[genre] += 1

        except Exception as e:
            print(f"  Failed {filename}: {e}")

        time.sleep(0.2)

print("\n================================")
print("DOWNLOAD COMPLETED")
print("================================")

total = 0

for genre in GENRES:
    print(f"{genre:12} : {counts[genre]} files")
    total += counts[genre]

print("--------------------------------")
print("TOTAL FILES:", total)
print("================================")