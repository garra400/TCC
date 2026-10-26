"""Confere tipo, dataset e alvo das tasks na API/cache oficial OpenML."""
import concurrent.futures
import hashlib
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import openml
import pandas as pd

from projeto.revisao.experimentos import PROJECT


def fetch(tid):
    try:
        path = Path(openml.config.get_cache_directory())/f"tasks/{tid}/task.xml"
        if path.exists():
            payload = path.read_bytes()
            origin = "openml_local_xml_cache"
        else:
            # ssl padrão usa a confiança do Windows; sem desativar verificação TLS.
            with urllib.request.urlopen(f"https://www.openml.org/api/v1/xml/task/{tid}",
                                        context=ssl.create_default_context(), timeout=25) as response:
                payload = response.read()
            origin = "openml_api_tls_verified"
        tree = ET.fromstring(payload)
        ns = {"o": "http://openml.org/openml"}
        did = int(tree.findtext(".//o:data_set_id", namespaces=ns))
        return {"task_id": tid, "task_type_id": int(tree.findtext("o:task_type_id", namespaces=ns)),
                "dataset_id": did, "target_name": tree.findtext(".//o:target_feature", namespaces=ns),
                "metadata_status": "ok", "metadata_source": origin,
                "xml_sha256": hashlib.sha256(payload).hexdigest(),
                "openml_dataset_url": f"https://www.openml.org/d/{did}"}
    except Exception as exc:
        return {"task_id": tid, "metadata_status": "error", "metadata_error": str(exc)}


if __name__ == "__main__":
    import numpy as np
    ids = np.load(PROJECT/"data/meta_conjunto/task_ids.npy").tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        frame = pd.DataFrame(pool.map(fetch, ids))
    output = PROJECT/"resultados/revisao_2026"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/"openml_tasks.csv", index=False)
    if frame.metadata_status.ne("ok").any():
        raise RuntimeError("Metadados incompletos; consulte openml_tasks.csv.")
    print(frame.groupby(["metadata_status", "task_type_id"]).size().to_string())
