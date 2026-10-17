"""
Mesin Penalaran Forward Chaining untuk Rekomendasi Pertolongan Pertama (P3K)
Memadukan fakta masukan dari deteksi visual YOLOv8 ke basis aturan R001-R005.
"""

import os
from typing import Dict, Any, List, Optional
import yaml

class ForwardChainingEngine:
    def __init__(self, rules_path: Optional[str] = None):
        if rules_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(base_dir, "rules.yaml")
        self.rules_path = rules_path
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if os.path.exists(self.rules_path):
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("rules", {})
        return {}

    @staticmethod
    def normalize_class_name(raw_name: str) -> str:
        s = raw_name.strip().lower()
        if "abrasion" in s or "lecet" in s:
            return "R001"
        elif "bruise" in s or "memar" in s:
            return "R002"
        elif "burn" in s or "bakar" in s or "blister" in s:
            return "R003"
        elif "cut" in s or "sayat" in s:
            return "R004"
        elif "abnormality" in s or "normal" in s:
            return "R005"
        return "R005"

    def evaluate(self, detection: Dict[str, Any], threshold_status: str = "detected") -> Dict[str, Any]:
        """
        Evaluasi fakta deteksi dengan mekanisme penalaran Forward Chaining.
        Fakta:
        - Jenis luka (class_name)
        - Tingkat keyakinan (confidence)
        - Status ambang batas: 'detected' (>=35%), 'possible' (29%-35%), 'not_detected' (<29%)
        - Estimasi rasio luas luka (area_ratio)
        """
        class_name = detection.get("class_name", "No Abnormality")
        conf = float(detection.get("confidence", 0.0))
        area_ratio = float(detection.get("area_ratio", 0.0))
        bbox = detection.get("bbox", [0, 0, 0, 0])

        rule_id = self.normalize_class_name(class_name)
        rule_data = self.rules.get(rule_id, {})

        label_id = rule_data.get("label_id", class_name)
        label_en = rule_data.get("label_en", class_name)
        base_priority = rule_data.get("base_priority", "Rendah")
        priority_level = rule_data.get("priority_level", 1)
        actions = list(rule_data.get("actions", []))
        critical_warnings = list(rule_data.get("critical_warnings", []))
        notes = rule_data.get("notes", "")

        # Aturan penyesuaian keparahan berdasarkan fakta luas area (Forward Chaining Fact Addition)
        reasoning_steps = [
            f"Fakta 1: Model YOLOv8 mengidentifikasi pola lesi kulit '{label_en}' dengan tingkat keyakinan {conf * 100:.1f}%.",
            f"Fakta 2: Status ambang batas deteksi berada pada kategori '{threshold_status.upper()}'.",
            f"Fakta 3: Rasio luas area lesi terhadap bidang pandang kamera adalah {area_ratio * 100:.2f}%."
        ]

        # Penyesuaian keparahan dinamis
        final_priority = base_priority
        if rule_id == "R003":  # Luka bakar
            if area_ratio > 0.15:
                final_priority = "Tinggi"
                priority_level = 3
                reasoning_steps.append("Fakta 4: Luas luka bakar melebihi 15% dari bidang pemotretan, memicu peningkatan prioritas ke 'Tinggi'.")
            elif threshold_status == "possible":
                final_priority = "Sedang"
        elif rule_id == "R004":  # Luka sayat
            if area_ratio > 0.12:
                final_priority = "Tinggi"
                priority_level = 3
                reasoning_steps.append("Fakta 4: Bounding box luka sayat terindikasi luas atau panjang, memicu peningkatan prioritas ke 'Tinggi'.")
        elif rule_id == "R005":
            final_priority = "Normal"
            priority_level = 0
            reasoning_steps.append("Fakta 4: Permukaan kulit tidak menampilkan lesi patologis aktif.")

        conclusion = (
            f"Berdasarkan rantai fakta tersebut, sistem mengaktifkan aturan {rule_id} ({label_id}). "
            f"Tingkat prioritas ditetapkan '{final_priority}', dan protokol pertolongan pertama {label_id} direkomendasikan."
        )

        return {
            "rule_id": rule_id,
            "label_id": label_id,
            "label_en": label_en,
            "confidence": conf,
            "threshold_status": threshold_status,
            "priority": final_priority,
            "priority_level": priority_level,
            "actions": actions,
            "critical_warnings": critical_warnings,
            "notes": notes,
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion,
            "bbox": bbox,
            "area_ratio": area_ratio
        }
