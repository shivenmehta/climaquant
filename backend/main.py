"""
main.py
ClimaQuant FastAPI backend.
Run with: uvicorn main:app --reload --port 8000
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from pricing.vanilla import calculate_vanilla_price, get_all_sigmas, COMPANIES, TICKER_MAP
from pricing.climate import calculate_climate_price
from pricing.bs_core import black_scholes_call
from data.loader import get_current_day_data, get_all_companies_data, build_research_payload
from climate.climate_model import get_regression_summary

app = FastAPI(
    title="ClimaQuant API",
    description="Climate-adjusted options pricing — Hacklytics 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_research_cache: dict = {}
_dashboard_cache: list = []
_sigma_cache: dict = {}
_ready = False


def _precompute():
    global _research_cache, _dashboard_cache, _sigma_cache, _ready
    print("ClimaQuant API starting up — pre-computing caches...")
    try:
        _dashboard_cache = get_all_companies_data()
        print(f"  ✓ Dashboard: {len(_dashboard_cache)} companies loaded")
    except Exception as e:
        print(f"  ✗ Dashboard cache failed: {e}")
    try:
        _sigma_cache = get_all_sigmas()
        print(f"  ✓ Sigmas cached for {len(_sigma_cache)} companies")
    except Exception as e:
        print(f"  ✗ Sigma cache failed: {e}")
    try:
        _research_cache = build_research_payload()
        print(f"  ✓ Research payload built")
    except Exception as e:
        print(f"  ✗ Research cache failed: {e}")
    _ready = True
    print("ClimaQuant API ready.")


@app.on_event("startup")
async def startup():
    print("ClimaQuant API ready — computing on demand.")
    global _ready
    _ready = True


class VanillaRequest(BaseModel):
    company:  str   = Field(..., example="apple")
    strike:   float = Field(..., example=185.0)
    maturity: float = Field(..., example=0.25)
    r:        float = Field(0.05)


class ClimateRequest(BaseModel):
    company:  str            = Field(..., example="apple")
    strike:   float          = Field(..., example=185.0)
    maturity: float          = Field(..., example=0.25)
    r:        float          = Field(0.05)
    hdd:      Optional[float] = Field(None)
    cdd:      Optional[float] = Field(None)
    palmer_z: Optional[float] = Field(None)


class CustomRequest(BaseModel):
    S:     float = Field(..., example=183.42)
    K:     float = Field(..., example=185.0)
    r:     float = Field(0.05)
    T:     float = Field(..., example=0.25)
    sigma: float = Field(..., example=0.23)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "api": "ClimaQuant v1.0", "ready": _ready}


@app.get("/api/companies", tags=["Meta"])
def list_companies():
    return [{"key": c, "ticker": TICKER_MAP.get(c, c.upper())} for c in COMPANIES]


@app.post("/api/vanilla-price", tags=["Pricing"])
def vanilla_price(req: VanillaRequest):
    if req.company not in COMPANIES:
        raise HTTPException(404, f"Unknown company '{req.company}'")
    try:
        return calculate_vanilla_price(req.company, req.strike, req.maturity, req.r)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pricing error: {e}")


@app.post("/api/climate-price", tags=["Pricing"])
def climate_price(req: ClimateRequest):
    if req.company not in COMPANIES:
        raise HTTPException(404, f"Unknown company '{req.company}'")
    try:
        return calculate_climate_price(
            req.company, req.strike, req.maturity, req.r,
            req.hdd, req.cdd, req.palmer_z
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Climate pricing error: {e}")


@app.get("/api/current-price", tags=["Dashboard"])
def current_price(company: Optional[str] = Query(None)):
    if company:
        if company not in COMPANIES:
            raise HTTPException(404, f"Unknown company '{company}'")
        for row in _dashboard_cache:
            if row.get("company") == company:
                return row
        try:
            return get_current_day_data(company)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, str(e))
    if _dashboard_cache:
        return _dashboard_cache
    return get_all_companies_data()


@app.post("/api/custom-price", tags=["Calculator"])
def custom_price(req: CustomRequest):
    return black_scholes_call(req.S, req.K, req.r, req.sigma, req.T)


@app.get("/api/research-data", tags=["Research"])
def research_data():
    if _research_cache:
        return _research_cache
    try:
        return build_research_payload()
    except Exception as e:
        raise HTTPException(500, f"Research computation failed: {e}")


@app.get("/api/sigmas", tags=["Meta"])
def all_sigmas():
    if _sigma_cache:
        return _sigma_cache
    return get_all_sigmas()


@app.get("/api/sigma-timeseries", tags=["Research"])
def sigma_timeseries(company: str = Query(...)):
    if company not in COMPANIES:
        raise HTTPException(404, f"Unknown company '{company}'")
    from climate.climate_model import get_sigma_timeseries
    result = get_sigma_timeseries(company)
    if result is None:
        raise HTTPException(503, "Climate data not available — ensure ClimateData/ CSVs are present")
    return result


@app.get("/api/scatter-3d", tags=["Research"])
def scatter_3d():
    from climate.climate_model import get_scatter_3d_data
    return get_scatter_3d_data()


@app.get("/api/debug", tags=["Debug"])
def debug_info():
    from pathlib import Path
    from climate.climate_model import CLIMATE_DIR, _CLIMATE_DATA_AVAILABLE, COMPANIES
    candidates = [
        Path(__file__).parent / "ClimateData",
        Path(__file__).parent.parent / "ClimateData",
        Path.cwd() / "ClimateData",
        Path.cwd() / "backend" / "ClimateData",
    ]
    return {
        "cwd":                    str(Path.cwd()),
        "main_py_dir":            str(Path(__file__).parent),
        "climate_dir_resolved":   str(CLIMATE_DIR),
        "climate_data_available": _CLIMATE_DATA_AVAILABLE,
        "candidates_checked":     [{"path": str(p), "exists": p.exists()} for p in candidates],
        "companies_count":        len(COMPANIES),
        "api_ready":              _ready,
    }