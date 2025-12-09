import numpy as np
from typing import List, Dict
from RecommendationFiles.recommendation_engine import FashionRecommendationEngine


class OutfitRecommender:
    def __init__(
        self,
        engine_pkl_path: str = "RecommendationFiles/recommendation_engine.pkl",
    ):
        # Load FAISS index + embeddings
        self.engine = FashionRecommendationEngine.load(engine_pkl_path)

    async def recommend_outfits(self, occasion: str, season: str, db, k: int = 10):
        """
        Build outfit recommendations using REAL wardrobe items stored in PostgreSQL.
        """
        wardrobe = await db.fetch(
            """
            SELECT w.item_id, w.category, w.metadata, e.embedding
            FROM wardrobe_items w
            JOIN embeddings e ON e.item_id = w.item_id
            """
        )

        if not wardrobe:
            return []

        def extract_list(meta, key):
            if not meta:
                return []
            if isinstance(meta, str):
                try:
                    import json
                    meta = json.loads(meta)
                except:
                    return []
            v = meta.get(key, [])
            if isinstance(v, str):
                return [v.lower()]
            return [x.lower() for x in v]

        filtered = []
        for row in wardrobe:
            meta = row["metadata"]
            item_occasions = extract_list(meta, "occasions")
            item_seasons = extract_list(meta, "season")

            if occasion.lower() in item_occasions and season.lower() in item_seasons:
                filtered.append(row)

        if not filtered:
            return []

        emb_matrix = np.vstack([row["embedding"] for row in filtered])
        query_vec = np.mean(emb_matrix, axis=0)

        recommendations = self.engine.recommend(
            query_embedding=query_vec,
            k=50,     # get a rich pool; we will filter by category next
        )

        tops, bottoms, shoes = [], [], []

        # Build lookup for speed
        wardrobe_map = {row["item_id"]: row for row in wardrobe}

        for rec in recommendations:
            idx = rec["index"]

            # model may index embeddings array, so map index → item_id
            if idx not in wardrobe_map:
                continue

            item = wardrobe_map[idx]
            cat = item["category"]

            if cat == "top":
                tops.append(idx)
            elif cat == "bottom":
                bottoms.append(idx)
            elif cat == "shoes":
                shoes.append(idx)

        # Must have at least one of each to build outfits
        if not tops or not bottoms or not shoes:
            return []

        outfits = []
        for t in tops[:5]:
            for b in bottoms[:5]:
                for s in shoes[:5]:
                    outfits.append({
                        "items": [t, b, s],
                        "score": 1.0
                    })

        return outfits[:k]
