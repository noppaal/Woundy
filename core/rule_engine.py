"""
Rule-Based System untuk Rekomendasi Pertolongan Pertama (First Aid)
Berdasarkan Mekanisme Forward Chaining (Tabel 3.1 Proposal Tugas Akhir)
Ahmad Naufal Ramadhan - Universitas Telkom 2026
"""

from typing import List, Dict, Any, Optional

RULE_DATABASE = {
    "R001": {
        "id": "R001",
        "name": "Abrasion",
        "label_id": "Luka Lecet (Abrasion)",
        "condition_desc": "Terdeteksi luka abrasi ringan, tidak ada perdarahan aktif",
        "base_priority": "Rendah",
        "priority_level": 1,
        "actions": [
            "Bersihkan area luka dengan air mengalir secara perlahan untuk menghilangkan kotoran atau partikel kecil.",
            "Oleskan antiseptik ringan (povidone iodine) secara merata di sekeliling permukaan luka.",
            "Tutup luka dengan perban steril atau plester pelekat guna melindungi luka dari kontaminasi bakteri.",
            "Ganti perban setiap 1-2 hari sekali atau segera bila perban basah dan kotor."
        ],
        "notes": "Tidak perlu ke dokter apabila luka lecet bersih dan sembuh tanpa tanda infeksi.",
        "warnings": [
            "Hindari menggosok permukaan luka secara agresif yang dapat merusak jaringan granulasi baru.",
            "Jangan gunakan alkohol pekat langsung pada luka terbuka karena dapat menimbulkan iritasi jaringan sel."
        ],
        "questions": [
            {
                "id": "abrasion_dirt",
                "text": "Apakah terdapat serpihan kerikil/kaca yang menancap dalam?",
                "options": ["Tidak ada", "Ada kotoran sulit dibersihkan"],
                "trigger_high_severity": True
            }
        ]
    },
    "R002": {
        "id": "R002",
        "name": "Bruise",
        "label_id": "Luka Memar (Bruise / Contusion)",
        "condition_desc": "Terdeteksi memar/contusion tanpa luka terbuka",
        "base_priority": "Rendah",
        "priority_level": 1,
        "actions": [
            "Kompres area memar dengan kantong es yang dibungkus kain bersih selama 15-20 menit.",
            "Istirahatkan (Rest) bagian tubuh yang mengalami benturan dan hindari beban berlebih.",
            "Elevasi (tinggikan) area yang terkena di atas posisi jantung bila memungkinkan untuk mengurangi pembengkakan.",
            "Jika timbul nyeri hebat yang tidak mereda atau area memar membesar signifikan, segera periksakan ke dokter."
        ],
        "notes": "Dapat dilakukan perawatan mandiri di rumah menggunakan protokol RICE (Rest, Ice, Compression, Elevation).",
        "warnings": [
            "JANGAN menempelkan es batu secara langsung pada kulit tanpa kain pembungkus karena dapat merusak jaringan (ice burn).",
            "JANGAN memijat atau mengurut area yang baru saja memar dalam 48 jam pertama."
        ],
        "questions": [
            {
                "id": "bruise_mobility",
                "text": "Apakah disertai ketidakmampuan menggerakkan sendi/tulang di sekitarnya?",
                "options": ["Bisa digerakkan normal", "Nyeri ekstrem / Tidak bisa digerakkan (dugaan fraktur)"],
                "trigger_high_severity": True
            }
        ]
    },
    "R003": {
        "id": "R003",
        "name": "Burn",
        "label_id": "Luka Bakar (Burn)",
        "condition_desc": "Terdeteksi luka bakar superfisial atau dengan lepuhan",
        "base_priority": "Rendah s.d. Tinggi",
        "sub_rules": {
            "mild": {
                "subtype": "Derajat 1 (Ringan - Kemerahan & Tanpa Lepuhan)",
                "priority": "Rendah",
                "priority_level": 1,
                "actions": [
                    "Siram area luka dengan air dingin mengalir (suhu ruangan) selama minimal 10-20 menit.",
                    "JANGAN gunakan es batu, pasta gigi, minyak goreng, atau mentega pada area luka bakar.",
                    "Oleskan gel lidah buaya murni (aloe vera) atau losion penyejuk khusus luka bakar ringan setelah dikeringkan perlahan."
                ],
                "notes": "Luka bakar ringan pada lapisan epidermis umumnya sembuh dalam beberapa hari dengan perawatan higienis."
            },
            "severe": {
                "subtype": "Derajat 2 / Parah (Ada Lepuhan / Bula atau Luas > 5 cm)",
                "priority": "Tinggi",
                "priority_level": 3,
                "actions": [
                    "Siram area luka secara hati-hati dengan air bersih mengalir untuk mendinginkan panas jaringan.",
                    "JANGAN memecahkan gelembung lepuhan (bula) secara sengaja karena meningkatkan risiko infeksi sekunder berat.",
                    "Tutup secara longgar dengan kain steril kering atau plastik wrap bersih tanpa ditekan.",
                    "Segera bawa korban ke fasilitas pelayanan kesehatan / IGD terdekat untuk evaluasi medis profesional."
                ],
                "notes": "Membutuhkan evaluasi dokter untuk pencegahan dehidrasi jaringan dan infeksi bakteri."
            }
        },
        "warnings": [
            "BAHAYA BESAR: Menempelkan pasta gigi, kecap, mentega, atau kopi pada luka bakar akan memperparah kerusakan jaringan dan menimbulkan infeksi mikrobial!",
            "Jangan gunakan es batu beku langsung karena dapat menyebabkan radang dingin (frostbite)."
        ],
        "questions": [
            {
                "id": "burn_blister_or_large",
                "text": "Apakah terdapat gelembung berisi cairan (lepuhan) ATAU luas luka bakar lebih besar dari 5 cm?",
                "options": ["Hanya kemerahan (Ringan)", "Ada lepuhan / Area luas (> 5 cm)"],
                "default": "Hanya kemerahan (Ringan)"
            }
        ]
    },
    "R004": {
        "id": "R004",
        "name": "Cut",
        "label_id": "Luka Sayat / Robek / Tusuk (Cut)",
        "condition_desc": "Terdeteksi luka akibat benda tajam, robekan, atau tusukan",
        "base_priority": "Sedang s.d. Tinggi",
        "sub_rules": {
            "mild": {
                "subtype": "Luka Sayat Ringan (Panjang < 2 cm & Dangkal)",
                "priority": "Sedang",
                "priority_level": 2,
                "actions": [
                    "Bersihkan luka dengan air bersih mengalir dan cairan antiseptik ringan.",
                    "Rapatkan kedua tepi luka dengan plester butterfly (kupu-kupu) atau plester strip steril.",
                    "Tutup luka dengan perban steril agar terlindung dari kotoran."
                ],
                "notes": "Dapat dirawat mandiri dengan pengawasan kebersihan harian."
            },
            "severe": {
                "subtype": "Luka Sayat Dalam / Luka Tusuk (Panjang > 2 cm / Perdarahan Aktif)",
                "priority": "Tinggi",
                "priority_level": 3,
                "actions": [
                    "Tekan luka kuat-kuat dengan kain kasa steril atau kain bersih tebal untuk menghentikan perdarahan.",
                    "JANGAN cabut benda tajam/asing jika masih tertancap dalam tubuh! Berikan fiksasi bantalan di sekeliling benda tersebut.",
                    "Balut dengan perban tekan yang mantap (pastikan tidak mematikan sirkulasi darah ujung jari).",
                    "Segera bawa ke Instalasi Gawat Darurat (IGD) atau fasilitas kesehatan terdekat untuk tindakan penjahitan / ATS tetanus."
                ],
                "notes": "Kondisi kegawatdaruratan: Penekanan langsung adalah kunci utama mencegah syok hipovolemik."
            }
        },
        "warnings": [
            "JANGAN PERNAH mencabut pisau, kaca, atau paku yang menancap karena benda tersebut berfungsi menahan perdarahan vaskular utama!",
            "Jika darah mengucur deras atau menyemprot seirama denyut nadi, segera hubungi 119/112."
        ],
        "questions": [
            {
                "id": "cut_deep_or_bleeding",
                "text": "Apakah luka terbuka dalam (> 2 cm), darah terus mengucur deras, atau terdapat benda menancap?",
                "options": ["Luka dangkal & perdarahan berhenti", "Luka dalam / Darah deras / Benda menancap"],
                "default": "Luka dangkal & perdarahan berhenti"
            }
        ]
    },
    "R005": {
        "id": "R005",
        "name": "No Abnormality",
        "label_id": "Tanpa Kelainan (Kulit Normal)",
        "condition_desc": "Hasil deteksi tidak menunjukkan adanya luka terbuka, memar, atau luka bakar",
        "base_priority": "N/A",
        "priority_level": 0,
        "actions": [
            "Kulit tampak normal dan sehat. Tidak ditemukan indikasi luka terbuka, memar, atau luka bakar.",
            "Rekomendasikan pengguna untuk tetap menjaga kebersihan dan hidrasi kulit secara rutin.",
            "Gunakan pelindung tabir surya (sunscreen) saat beraktivitas di bawah paparan sinar matahari langsung."
        ],
        "notes": "Kontrol negatif normal: tidak memerlukan tindakan penanganan darurat.",
        "warnings": [
            "Apabila merasakan nyeri internal hebat atau mati rasa meskipun tanpa tampak luka, konsultasikan dengan tenaga medis."
        ],
        "questions": []
    }
}

class ForwardChainingEngine:
    """
    Inference Engine berbasis Forward Chaining
    Menelusuri fakta deteksi citra (jenis luka, luas bounding box, confidence, atribut pengguna)
    menuju aksi pertolongan pertama (Tabel 3.1).
    """

    def __init__(self):
        self.rules = RULE_DATABASE

    def _normalize_class_name(self, raw_name: str) -> str:
        s = raw_name.strip().lower()
        if "abrasion" in s:
            return "R001"
        elif "bruise" in s:
            return "R002"
        elif "burn" in s or "blister" in s:
            return "R003"
        elif "cut" in s:
            return "R004"
        elif "abnormality" in s or "normal" in s:
            return "R005"
        return "R005"

    def evaluate_detection(
        self,
        detected_objects: List[Dict[str, Any]],
        user_answers: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Menjalankan Forward Chaining pada himpunan objek terdeteksi.
        """
        user_answers = user_answers or {}
        
        # Jika tidak ada objek terdeteksi sama sekali, picu aturan R005
        if not detected_objects:
            detected_objects = [{
                "class_name": "No Abnormality",
                "confidence": 0.95,
                "bbox": [0, 0, 100, 100],
                "area_ratio": 1.0
            }]

        fired_rules = []
        overall_priority_level = 0
        priority_label_map = {0: "N/A", 1: "Rendah", 2: "Sedang", 3: "Tinggi", 4: "Gawat Darurat"}

        for obj in detected_objects:
            raw_cls = obj.get("class_name", "No Abnormality")
            conf = float(obj.get("confidence", 0.0))
            area_ratio = float(obj.get("area_ratio", 0.0))
            bbox = obj.get("bbox", [])
            rule_id = self._normalize_class_name(raw_cls)
            rule = self.rules[rule_id]

            evaluation_item = {
                "rule_id": rule_id,
                "wound_name": rule["name"],
                "label_id": rule["label_id"],
                "condition_matched": rule["condition_desc"],
                "confidence": conf,
                "bbox": bbox,
                "area_ratio": area_ratio,
                "warnings": rule.get("warnings", []),
                "questions": rule.get("questions", [])
            }

            # Evaluasi Forward Chaining per jenis aturan
            if rule_id == "R001":
                # Abrasion
                has_deep_dirt = user_answers.get("abrasion_dirt", False)
                p_level = 2 if has_deep_dirt else 1
                evaluation_item["priority"] = "Sedang" if has_deep_dirt else "Rendah"
                evaluation_item["priority_level"] = p_level
                evaluation_item["actions"] = rule["actions"][:]
                if has_deep_dirt:
                    evaluation_item["actions"].append(
                        "PERINGATAN: Serpihan benda sulit dibersihkan memerlukan irigasi steril oleh tenaga medis untuk mencegah infeksi tetanus."
                    )
                evaluation_item["notes"] = rule["notes"]

            elif rule_id == "R002":
                # Bruise
                mobility_problem = user_answers.get("bruise_mobility", False)
                p_level = 3 if mobility_problem else 1
                evaluation_item["priority"] = "Tinggi" if mobility_problem else "Rendah"
                evaluation_item["priority_level"] = p_level
                evaluation_item["actions"] = rule["actions"][:]
                if mobility_problem:
                    evaluation_item["actions"].append(
                        "URGENT: Dugaan fraktur/cedera ligamen berat! Segera pasang bidai/penyangga sementara dan bawa ke faskes untuk rontgen."
                    )
                evaluation_item["notes"] = rule["notes"]

            elif rule_id == "R003":
                # Burn: Cek apakah area luas > 15% atau user menjawab ada lepuhan
                user_blister = user_answers.get("burn_blister_or_large", None)
                # Estimasi dari computer vision: jika bbox mencakup > 15% citra
                is_large_by_vision = area_ratio > 0.15 or "blister" in raw_cls.lower()
                
                if user_blister is True or (user_blister is None and is_large_by_vision):
                    sub = rule["sub_rules"]["severe"]
                else:
                    sub = rule["sub_rules"]["mild"]

                evaluation_item["subtype"] = sub["subtype"]
                evaluation_item["priority"] = sub["priority"]
                evaluation_item["priority_level"] = sub["priority_level"]
                evaluation_item["actions"] = sub["actions"]
                evaluation_item["notes"] = sub["notes"]

            elif rule_id == "R004":
                # Cut: Cek apakah luka dalam atau darah mengucur
                user_severe_cut = user_answers.get("cut_deep_or_bleeding", None)
                is_large_by_vision = area_ratio > 0.12
                
                if user_severe_cut is True or (user_severe_cut is None and is_large_by_vision):
                    sub = rule["sub_rules"]["severe"]
                else:
                    sub = rule["sub_rules"]["mild"]

                evaluation_item["subtype"] = sub["subtype"]
                evaluation_item["priority"] = sub["priority"]
                evaluation_item["priority_level"] = sub["priority_level"]
                evaluation_item["actions"] = sub["actions"]
                evaluation_item["notes"] = sub["notes"]

            elif rule_id == "R005":
                # Normal
                evaluation_item["priority"] = "N/A"
                evaluation_item["priority_level"] = 0
                evaluation_item["actions"] = rule["actions"]
                evaluation_item["notes"] = rule["notes"]

            overall_priority_level = max(overall_priority_level, evaluation_item["priority_level"])
            fired_rules.append(evaluation_item)

        # Urutkan berdasarkan prioritas tertinggi terlebih dahulu
        fired_rules.sort(key=lambda x: x["priority_level"], reverse=True)

        overall_priority_text = priority_label_map.get(overall_priority_level, "Rendah")
        is_emergency = overall_priority_level >= 3

        # Multi-wound synthesis
        is_multi_wound = len(fired_rules) > 1 and fired_rules[0]["rule_id"] != "R005"
        summary_advice = []
        if is_multi_wound:
            summary_advice.append(
                f"Terdeteksi {len(fired_rules)} jenis luka secara bersamaan. "
                f"Prioritaskan penanganan luka dengan tingkat kegawatan tertinggi terlebih dahulu ({fired_rules[0]['label_id']})."
            )

        return {
            "success": True,
            "overall_priority": overall_priority_text,
            "overall_priority_level": overall_priority_level,
            "is_emergency": is_emergency,
            "is_multi_wound": is_multi_wound,
            "summary_advice": summary_advice,
            "fired_rules": fired_rules,
            "emergency_contact": {
                "ambulance": "119",
                "national_emergency": "112",
                "red_cross_indonesia": "021-7992325"
            }
        }
