import { useEffect, useMemo, useState } from "react";

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Database,
  History,
  MapPin,
  Navigation,
  RefreshCw,
  Satellite,
  Ship,
  Upload,
  Wind,
  Waves,
  X,
  Wifi,
  WifiOff,
  Server,
} from "lucide-react";

import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  Polyline,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// DEVELOPMENT FALLBACK DATA
// ============================================================

const DEMO_VESSELS = [
  {
    vessel_name: "MV Test Vessel",
    vessel_type: "TANKER",
    latitude: 25.8,
    longitude: 54.65,
    speed_knots: 8.5,
    heading_degrees: 180,
  },
  {
    vessel_name: "MV Ocean Carrier",
    vessel_type: "CARGO",
    latitude: 25.7,
    longitude: 54.68,
    speed_knots: 12,
    heading_degrees: 90,
  },
  {
    vessel_name: "MV Chemical Star",
    vessel_type: "CHEMICAL TANKER",
    latitude: 25.75,
    longitude: 54.66,
    speed_knots: 6,
    heading_degrees: 270,
  },
];

const DEMO_CANDIDATES = [
  {
    candidate_id: "C1",
    classification: "PROBABLE OIL",
    oil_confidence: 0.7,
    validation_score: 70,
    location: {
      latitude: 25.824077,
      longitude: 54.649044,
    },
    area_pixels: 8169,
    area_ratio: 0.00195,
    dimensions: {
      width_pixels: 160,
      height_pixels: 162,
    },
    aspect_ratio: 1.01,
    compactness: 0.042,
    orientation: -29.6,
    mean_backscatter: 56.17,
    darkness_score: 1.426,
    ais_vessels: [
      {
        vessel_name: "MV Test Vessel",
        vessel_type: "TANKER",
        distance_to_spill_km: 2.68,
        speed_knots: 8.5,
      },
    ],
    wind_analysis: {
      alignment_score: 50,
      classification: "MODERATE",
    },
    current_analysis: {
      alignment_score: 50,
      classification: "MODERATE",
    },
  },
  {
    candidate_id: "C2",
    classification: "REQUIRES REVIEW",
    oil_confidence: 0.65,
    validation_score: 65,
    location: {
      latitude: 25.698777,
      longitude: 54.679315,
    },
    area_pixels: 5954,
    area_ratio: 0.00142,
    dimensions: {
      width_pixels: 102,
      height_pixels: 208,
    },
    aspect_ratio: 2.04,
    compactness: 0.039,
    orientation: -88.39,
    mean_backscatter: 56.27,
    darkness_score: 1.424,
    ais_vessels: [
      {
        vessel_name: "MV Ocean Carrier",
        vessel_type: "CARGO",
        distance_to_spill_km: 0.15,
        speed_knots: 12,
      },
    ],
    wind_analysis: {
      alignment_score: 100,
      classification: "STRONG ALIGNMENT",
    },
    current_analysis: {
      alignment_score: 100,
      classification: "STRONG ALIGNMENT",
    },
  },
  {
    candidate_id: "C3",
    classification: "REQUIRES REVIEW",
    oil_confidence: 0.5,
    validation_score: 50,
    location: {
      latitude: 25.752003,
      longitude: 54.661028,
    },
    area_pixels: 4479,
    area_ratio: 0.00107,
    dimensions: {
      width_pixels: 110,
      height_pixels: 127,
    },
    aspect_ratio: 1.15,
    compactness: 0.040,
    orientation: -88.09,
    mean_backscatter: 53.31,
    darkness_score: 1.497,
    ais_vessels: [
      {
        vessel_name: "MV Chemical Star",
        vessel_type: "CHEMICAL TANKER",
        distance_to_spill_km: 0.25,
        speed_knots: 6,
      },
    ],
    wind_analysis: {
      alignment_score: 100,
      classification: "STRONG ALIGNMENT",
    },
    current_analysis: {
      alignment_score: 100,
      classification: "STRONG ALIGNMENT",
    },
  },
];


// ============================================================
// HELPERS
// ============================================================

function normalizeCandidate(candidate, index = 0) {
  const geometry = candidate?.geometry || {};
  const bbox = geometry?.bbox || {};
  const detector = candidate?.detector || {};

  const sarRegionConfidence = Number(
    candidate?.sar_region_confidence ??
    detector?.mean_confidence ??
    candidate?.oil_confidence ??
    0
  );

  const classifierConfidence = Number(
    candidate?.classification_confidence ??
    candidate?.classifier?.confidence ??
    0
  );

  return {
    ...candidate,

    candidate_id:
      candidate?.candidate_id ||
      candidate?.id ||
      `C${index + 1}`,

    classification:
      candidate?.classification ||
      "REQUIRES REVIEW",

    // Keep the backend's actual model outputs available to the UI.
    classification_confidence: classifierConfidence,
    classifier_confidence: classifierConfidence,
    sar_region_confidence: sarRegionConfidence,
    oil_confidence: sarRegionConfidence,

    validation_score:
      Number(candidate?.validation_score ?? 0),

    geometry,
    area_pixels:
      candidate?.area_pixels ??
      geometry?.area_pixels ??
      null,
    area_ratio:
      candidate?.area_ratio ?? null,
    dimensions:
      candidate?.dimensions ||
      (bbox?.width != null && bbox?.height != null
        ? {
            width_pixels: Number(bbox.width),
            height_pixels: Number(bbox.height),
          }
        : null),

    location:
      candidate?.location &&
      Number.isFinite(Number(candidate.location.latitude)) &&
      Number.isFinite(Number(candidate.location.longitude))
        ? {
            latitude: Number(candidate.location.latitude),
            longitude: Number(candidate.location.longitude),
          }
        : null,

    ais_vessels:
      getCandidateVessels(candidate),

    wind_analysis:
      candidate?.wind_analysis || {},

    current_analysis:
      candidate?.current_analysis || {},

    backward_drift:
      candidate?.backward_drift || null,

    estimated_source:
      candidate?.estimated_source ||
      candidate?.backward_drift?.estimated_source ||
      null,

    source_uncertainty_km:
      Number(
        candidate?.source_uncertainty_km ??
        candidate?.backward_drift?.uncertainty_radius_km ??
        candidate?.backward_drift?.source_uncertainty_km ??
        NaN
      ),
  };
}



function normalizeVessel(vessel, index = 0) {
  if (!vessel) return null;

  const lat = Number(
    vessel?.latitude ??
    vessel?.lat ??
    vessel?.position?.latitude ??
    vessel?.position?.lat
  );

  const lon = Number(
    vessel?.longitude ??
    vessel?.lon ??
    vessel?.lng ??
    vessel?.position?.longitude ??
    vessel?.position?.lon ??
    vessel?.position?.lng
  );

  const distanceRaw =
    vessel?.distance_to_spill_km ??
    vessel?.distance_km ??
    vessel?.distanceKm ??
    vessel?.distance ??
    vessel?.proximity_km ??
    vessel?.proximity?.distance_km ??
    vessel?.proximity?.distance;

  const priorityRaw =
    vessel?.priority ??
    vessel?.priority_score ??
    vessel?.attribution?.priority_score ??
    vessel?.attribution?.score ??
    vessel?.attribution_score ??
    vessel?.correlation_score ??
    vessel?.score;

  return {
    ...vessel,

    vessel_id:
      vessel?.vessel_id ||
      vessel?.mmsi ||
      vessel?.imo ||
      vessel?.id ||
      `AIS-${index + 1}`,

    vessel_name:
      vessel?.vessel_name ||
      vessel?.name ||
      vessel?.ship_name ||
      `AIS Vessel ${index + 1}`,

    vessel_type:
      vessel?.vessel_type ||
      vessel?.ship_type ||
      vessel?.type ||
      "VESSEL",

    latitude: Number.isFinite(lat) ? lat : null,
    longitude: Number.isFinite(lon) ? lon : null,

    speed_knots: Number(
      vessel?.speed_knots ??
      vessel?.speed ??
      vessel?.sog ??
      0
    ),

    heading_degrees: Number(
      vessel?.heading_degrees ??
      vessel?.heading ??
      vessel?.course ??
      vessel?.cog ??
      0
    ),

    distance_to_spill_km:
      distanceRaw != null && Number.isFinite(Number(distanceRaw))
        ? Number(distanceRaw)
        : null,

    priority:
      priorityRaw != null && Number.isFinite(Number(priorityRaw))
        ? Number(priorityRaw)
        : null,

    correlation_score:
      vessel?.correlation_score ??
      vessel?.correlation ??
      vessel?.attribution?.score ??
      vessel?.attribution_score ??
      null,

    attribution:
      vessel?.attribution || null,
  };
}


function getCandidateVessels(candidate) {
  const sources = [
    candidate?.ais_vessels,
    candidate?.ais_contacts,
    candidate?.vessels,
    candidate?.ais_candidates,
    candidate?.ais?.vessels,
    candidate?.ais?.contacts,
    candidate?.ais_correlation?.vessels,
    candidate?.ais_correlation?.candidates,
  ];

  return sources
    .filter(Array.isArray)
    .flat()
    .map((vessel, index) => normalizeVessel(vessel, index))
    .filter(Boolean);
}


function collectAisVessels(data, candidates) {
  const collected = [];

  const topLevelSources = [
    data?.ais_vessels,
    data?.ais_contacts,
    data?.vessels,
    data?.ais_candidates,
    data?.ais?.vessels,
    data?.ais?.contacts,
    data?.ais_correlation?.vessels,
    data?.ais_correlation?.candidates,
  ];

  topLevelSources
    .filter(Array.isArray)
    .forEach((items) => {
      items.forEach((vessel) => {
        const normalized = normalizeVessel(
          vessel,
          collected.length
        );
        if (normalized) collected.push(normalized);
      });
    });

  (candidates || []).forEach((candidate) => {
    getCandidateVessels(candidate).forEach((vessel) => {
      collected.push(vessel);
    });
  });

  // De-duplicate the same vessel when it appears both at
  // top-level and inside a candidate.
  const seen = new Set();

  return collected.filter((vessel) => {
    const key =
      vessel?.mmsi ||
      vessel?.imo ||
      vessel?.vessel_id ||
      `${vessel?.vessel_name}|${vessel?.latitude}|${vessel?.longitude}`;

    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}


function getConfidence(candidate) {
  const value = Number(
    candidate?.sar_region_confidence ??
    candidate?.detector?.mean_confidence ??
    candidate?.oil_confidence ??
    candidate?.validation_score ??
    0
  );

  return value <= 1 ? value * 100 : value;
}


function getNearestVessel(candidate) {
  if (
    !candidate?.ais_vessels ||
    candidate.ais_vessels.length === 0
  ) {
    return null;
  }

  return [...candidate.ais_vessels].sort(
    (a, b) =>
      Number(a.distance_to_spill_km ?? 999999) -
      Number(b.distance_to_spill_km ?? 999999)
  )[0];
}


function getAlignmentScore(data) {
  if (!data) return null;

  const score =
    data.alignment_score ??
    data.score ??
    data.alignmentScore;

  if (score === undefined || score === null) {
    return null;
  }

  return Number(score);
}


function getBackwardDrift(candidate) {
  return (
    candidate?.backward_drift ||
    candidate?.drift_reconstruction ||
    candidate?.source_reconstruction ||
    null
  );
}


function normalizeGeoPoint(point) {
  if (!point) return null;

  const latitude = Number(
    point?.latitude ??
    point?.lat ??
    point?.position?.latitude ??
    point?.position?.lat
  );

  const longitude = Number(
    point?.longitude ??
    point?.lon ??
    point?.lng ??
    point?.position?.longitude ??
    point?.position?.lon ??
    point?.position?.lng
  );

  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return null;
  }

  return { latitude, longitude };
}


function getDriftTrajectory(candidate) {
  const drift = getBackwardDrift(candidate);
  if (!drift) return [];

  const raw =
    drift?.trajectory ||
    drift?.path ||
    drift?.backward_trajectory ||
    drift?.points ||
    [];

  if (!Array.isArray(raw)) return [];

  return raw
    .map((point) => {
      const geo = normalizeGeoPoint(point);
      if (!geo) return null;

      return {
        ...geo,
        hours: Number(
          point?.hours_before ??
          point?.hours ??
          point?.time_hours ??
          point?.elapsed_hours ??
          0
        ),
      };
    })
    .filter(Boolean);
}


function getEstimatedSource(candidate) {
  const drift = getBackwardDrift(candidate);

  const direct = normalizeGeoPoint(
    candidate?.estimated_source ||
    drift?.estimated_source ||
    drift?.source ||
    drift?.origin ||
    drift?.estimated_origin ||
    drift?.source_location
  );

  if (direct) return direct;

  // Backward drift trajectories are returned from T-0h (spill)
  // toward the oldest point (estimated source).
  const trajectory = getDriftTrajectory(candidate);

  if (trajectory.length > 0) {
    return normalizeGeoPoint(
      trajectory[trajectory.length - 1]
    );
  }

  return null;
}


function getSourceUncertainty(candidate) {
  const drift = getBackwardDrift(candidate);

  const value = Number(
    candidate?.source_uncertainty_km ??
    candidate?.uncertainty_radius_km ??
    drift?.uncertainty_radius_km ??
    drift?.source_uncertainty_km ??
    drift?.uncertainty_km
  );

  return Number.isFinite(value) ? value : null;
}


// ============================================================
// MAP CONTROLLER
// ============================================================

function MapController({ center, zoom = 7 }) {
  const map = useMap();

  useEffect(() => {
    if (
      center &&
      Number.isFinite(center[0]) &&
      Number.isFinite(center[1])
    ) {
      map.setView(center, zoom, {
        animate: true,
      });
    }
  }, [center, zoom, map]);

  return null;
}


// ============================================================
// PAGE HEADER
// ============================================================

function PageHeader() {
  return (
    <div className="page-header">
      <div>
        <div className="eyebrow">
          OPERATIONS / SPILL MONITORING
        </div>

        <h1>Maritime Spill Intelligence</h1>

        <p>
          Satellite-based detection and AIS source attribution
        </p>
      </div>

      <div className="session-info">
        <div className="operational-badge">
          <span className="status-dot" />
          OPERATIONAL
        </div>

        <div className="session-text">
          <strong>Development Session</strong>
          <small>
            Data sources: SAR / AIS / Environment
          </small>
        </div>
      </div>
    </div>
  );
}


// ============================================================
// NAV ITEM
// ============================================================

function NavItem({
  icon,
  label,
  active,
  onClick,
}) {
  return (
    <button
      className={`nav-item ${active ? "active" : ""}`}
      onClick={onClick}
      type="button"
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}


// ============================================================
// SIDEBAR
// ============================================================

function Sidebar({
  activePage,
  setActivePage,
  backendOnline,
}) {
  return (
    <aside className="sidebar">

      <div className="brand">
        <div className="brand-icon">
          <Satellite size={24} />
        </div>

        <div>
          <div className="brand-name">
            SPILLTRACE
          </div>

          <div className="brand-subtitle">
            MARITIME INTELLIGENCE
          </div>
        </div>
      </div>


      <div className="sidebar-content">

        <div className="nav-section-title">
          WORKSPACE
        </div>

        <NavItem
          icon={<Activity size={19} />}
          label="Overview"
          active={activePage === "overview"}
          onClick={() =>
            setActivePage("overview")
          }
        />

        <NavItem
          icon={<Satellite size={19} />}
          label="SAR Detections"
          active={activePage === "sar"}
          onClick={() =>
            setActivePage("sar")
          }
        />

        <NavItem
          icon={<Ship size={19} />}
          label="AIS Vessels"
          active={activePage === "ais"}
          onClick={() =>
            setActivePage("ais")
          }
        />

        <NavItem
          icon={<Wind size={19} />}
          label="Environment"
          active={activePage === "environment"}
          onClick={() =>
            setActivePage("environment")
          }
        />


        <div className="nav-section-title analysis-title">
          ANALYSIS
        </div>

        <NavItem
          icon={<History size={19} />}
          label="Detection History"
          active={activePage === "history"}
          onClick={() =>
            setActivePage("history")
          }
        />

        <NavItem
          icon={<Database size={19} />}
          label="Data Sources"
          active={activePage === "sources"}
          onClick={() =>
            setActivePage("sources")
          }
        />

        <NavItem
          icon={<Server size={19} />}
          label="System Status"
          active={activePage === "status"}
          onClick={() =>
            setActivePage("status")
          }
        />

      </div>


      <div className="sidebar-footer">

        <div className="system-status">

          {backendOnline ? (
            <CheckCircle2 size={17} />
          ) : (
            <WifiOff size={17} />
          )}

          <div>
            <strong>
              {backendOnline
                ? "System Online"
                : "Backend Offline"}
            </strong>

            <small>
              {backendOnline
                ? "API services available"
                : "Start FastAPI backend"}
            </small>
          </div>

        </div>

        <div className="version">
          SPILLTRACE v1.0
        </div>

      </div>

    </aside>
  );
}


// ============================================================
// UPLOAD PANEL
// ============================================================

function UploadPanel({
  selectedFile,
  setSelectedFile,
  onAnalyze,
  analyzing,
}) {
  return (
    <div className="upload-panel">

      <div className="upload-heading">

        <div className="eyebrow">
          INPUT
        </div>

        <h2>
          Analyze SAR Image
        </h2>

      </div>


      <label className="upload-zone">

        <input
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={(event) => {
            const file =
              event.target.files?.[0];

            if (file) {
              setSelectedFile(file);
            }
          }}
          hidden
        />

        <Upload size={34} />

        {selectedFile ? (
          <>
            <strong>
              {selectedFile.name}
            </strong>

            <span>
              {(selectedFile.size / 1024 / 1024).toFixed(
                2
              )}{" "}
              MB
            </span>
          </>
        ) : (
          <>
            <strong>
              Upload Sentinel SAR image
            </strong>

            <span>
              TIFF, PNG or JPEG
            </span>
          </>
        )}

      </label>


      <button
        className="analyze-button"
        type="button"
        disabled={!selectedFile || analyzing}
        onClick={onAnalyze}
      >
        {analyzing ? (
          <>
            <RefreshCw
              size={18}
              className="spin"
            />

            Analyzing...
          </>
        ) : (
          <>
            <Satellite size={18} />

            Run Spill Analysis
          </>
        )}
      </button>

    </div>
  );
}


// ============================================================
// STAT CARD
// ============================================================

function StatCard({
  icon,
  label,
  value,
  subtitle,
}) {
  return (
    <div className="stat-card">

      <div className="stat-icon">
        {icon}
      </div>

      <div className="stat-content">

        <span className="stat-label">
          {label}
        </span>

        <strong className="stat-value">
          {value}
        </strong>

        <small>
          {subtitle}
        </small>

      </div>

    </div>
  );
}


// ============================================================
// MAP LEGEND
// ============================================================

function MapLegend() {
  return (
    <div className="map-legend">

      <div className="eyebrow">
        MAP LEGEND
      </div>

      <div className="legend-row">
        <span className="legend-spill" />
        Spill candidate
      </div>

      <div className="legend-row">
        <span className="legend-vessel">
          ▲
        </span>
        AIS vessel
      </div>

      <div className="legend-row">
        <span className="legend-line" />
        Attribution link
      </div>

      <div className="legend-row">
        <span
          className="legend-line"
          style={{
            borderTop: "3px dashed #a855f7",
          }}
        />
        Backward drift / source
      </div>

    </div>
  );
}


// ============================================================
// MAIN MAP
// ============================================================

function DetectionMap({
  candidates,
  vessels,
  selectedCandidate,
  setSelectedCandidate,
}) {
  const firstCandidate =
    candidates?.[0];

  const center = firstCandidate?.location
    ? [
        firstCandidate.location.latitude,
        firstCandidate.location.longitude,
      ]
    : [25.75, 54.66];


  const selectedVessel =
    selectedCandidate
      ? getNearestVessel(selectedCandidate)
      : null;


  const polyline =
    selectedCandidate &&
    selectedVessel &&
    selectedCandidate.location
      ? [
          [
            selectedCandidate.location
              .latitude,
            selectedCandidate.location
              .longitude,
          ],
          [
            Number(
              selectedVessel.latitude ??
                selectedVessel.lat ??
                0
            ),
            Number(
              selectedVessel.longitude ??
                selectedVessel.lon ??
                0
            ),
          ],
        ]
      : null;


  const driftTrajectory =
    selectedCandidate
      ? getDriftTrajectory(selectedCandidate)
      : [];

  const estimatedSource =
    selectedCandidate
      ? getEstimatedSource(selectedCandidate)
      : null;

  const driftPolyline =
    driftTrajectory.length >= 2
      ? driftTrajectory.map((point) => [
          point.latitude,
          point.longitude,
        ])
      : null;


  return (
    <div className="map-panel">

      <div className="panel-header">

        <div>
          <div className="eyebrow">
            GEOSPATIAL ANALYSIS
          </div>

          <h2>
            Detection Area
          </h2>
        </div>

        <div className="map-controls">

          <span className="map-tag">
            SENTINEL SAR
          </span>

          <span className="map-tag">
            LIVE API
          </span>

          {selectedCandidate && (
            <span className="map-tag">
              FOCUS{" "}
              {selectedCandidate.candidate_id}
            </span>
          )}

        </div>

      </div>


      <div className="map-wrapper">

        <MapContainer
          center={center}
          zoom={7}
          scrollWheelZoom={true}
          className="leaflet-map"
        >

          <TileLayer
            attribution="Tiles © Esri"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />

          <MapController
            center={
              selectedCandidate?.location
                ? [
                    selectedCandidate.location
                      .latitude,
                    selectedCandidate.location
                      .longitude,
                  ]
                : center
            }
            zoom={7}
          />


          {candidates.map(
            (candidate, index) => {
              if (
                !candidate.location ||
                !Number.isFinite(
                  candidate.location.latitude
                ) ||
                !Number.isFinite(
                  candidate.location.longitude
                )
              ) {
                return null;
              }

              const isSelected =
                selectedCandidate?.candidate_id ===
                candidate.candidate_id;

              return (
                <CircleMarker
                  key={
                    candidate.candidate_id ||
                    index
                  }
                  center={[
                    candidate.location.latitude,
                    candidate.location.longitude,
                  ]}
                  radius={
                    isSelected ? 13 : 9
                  }
                  pathOptions={{
                    color: "#ff8a00",
                    fillColor:
                      isSelected
                        ? "#ffb000"
                        : "#ff7200",
                    fillOpacity: 0.9,
                    weight:
                      isSelected ? 4 : 2,
                  }}
                  eventHandlers={{
                    click: () =>
                      setSelectedCandidate(
                        candidate
                      ),
                  }}
                >
                  <Popup>
                    <strong>
                      {candidate.candidate_id}
                    </strong>

                    <br />

                    {candidate.classification}

                    <br />

                    Confidence:{" "}
                    {Math.round(
                      getConfidence(candidate)
                    )}
                    %
                  </Popup>
                </CircleMarker>
              );
            }
          )}


          {driftPolyline && (
            <Polyline
              positions={driftPolyline}
              pathOptions={{
                color: "#a855f7",
                weight: 4,
                opacity: 0.9,
                dashArray: "10 8",
              }}
            />
          )}


          {estimatedSource && (
            <CircleMarker
              center={[
                estimatedSource.latitude,
                estimatedSource.longitude,
              ]}
              radius={12}
              pathOptions={{
                color: "#a855f7",
                fillColor: "#7c3aed",
                fillOpacity: 0.95,
                weight: 3,
              }}
            >
              <Popup>
                <strong>
                  Estimated Source
                </strong>

                <br />

                {estimatedSource.latitude.toFixed(5)}° N, {estimatedSource.longitude.toFixed(5)}° E

                <br />

                Backward drift reconstruction

                {getSourceUncertainty(selectedCandidate) != null && (
                  <>
                    <br />
                    Uncertainty: {getSourceUncertainty(selectedCandidate).toFixed(1)} km
                  </>
                )}
              </Popup>
            </CircleMarker>
          )}


          {vessels.map(
            (vessel, index) => {
              const lat = Number(
                vessel.latitude ??
                  vessel.lat
              );

              const lon = Number(
                vessel.longitude ??
                  vessel.lon
              );

              if (
                !Number.isFinite(lat) ||
                !Number.isFinite(lon)
              ) {
                return null;
              }

              return (
                <CircleMarker
                  key={`vessel-${index}`}
                  center={[lat, lon]}
                  radius={10}
                  pathOptions={{
                    color: "#00d9ff",
                    fillColor: "#06151e",
                    fillOpacity: 1,
                    weight: 2,
                  }}
                >
                  <Popup>

                    <strong>
                      {vessel.vessel_name ||
                        vessel.name ||
                        "AIS Vessel"}
                    </strong>

                    <br />

                    {vessel.vessel_type ||
                      "VESSEL"}

                    <br />

                    Speed:{" "}
                    {vessel.speed_knots ??
                      "—"}{" "}
                    kn

                  </Popup>
                </CircleMarker>
              );
            }
          )}


          {polyline && (
            <Polyline
              positions={polyline}
              pathOptions={{
                color: "#ffae00",
                weight: 2,
                dashArray: "8 7",
              }}
            />
          )}

        </MapContainer>


        <MapLegend />

      </div>

    </div>
  );
}


// ============================================================
// CANDIDATE LIST
// ============================================================

function CandidateList({
  candidates,
  selectedCandidate,
  setSelectedCandidate,
}) {
  return (
    <div className="panel">

      <div className="panel-header">

        <div>
          <div className="eyebrow">
            SAR ANALYSIS
          </div>

          <h2>
            Detection Candidates
          </h2>
        </div>

        <div className="count-badge">
          {String(candidates.length).padStart(
            2,
            "0"
          )}
        </div>

      </div>


      {candidates.length === 0 ? (
        <div className="empty-state">

          <Satellite size={42} />

          <h3>
            No SAR detections
          </h3>

          <p>
            Upload a Sentinel SAR image and
            run the analysis.
          </p>

        </div>
      ) : (
        <div className="candidate-list">

          {candidates.map(
            (candidate, index) => {
              const nearest =
                getNearestVessel(
                  candidate
                );

              const confidence =
                getConfidence(candidate);

              const isSelected =
                selectedCandidate?.candidate_id ===
                candidate.candidate_id;

              return (
                <button
                  key={
                    candidate.candidate_id ||
                    index
                  }
                  type="button"
                  className={`candidate-row ${
                    isSelected
                      ? "selected"
                      : ""
                  }`}
                  onClick={() =>
                    setSelectedCandidate(
                      candidate
                    )
                  }
                >

                  <div className="candidate-main">

                    <span className="candidate-dot" />

                    <div>

                      <strong>
                        {candidate.candidate_id}
                      </strong>

                      <small
                        className={
                          candidate.classification ===
                          "OIL"
                            ? "probable"
                            : "review"
                        }
                      >
                        {candidate.classification}
                      </small>

                    </div>

                  </div>


                  <div className="candidate-location">

                    <MapPin size={14} />

                    <span>
                      {candidate.location
                        ? `${candidate.location.latitude.toFixed(
                            5
                          )}, ${candidate.location.longitude.toFixed(
                            5
                          )}`
                        : "—"}
                    </span>

                  </div>


                  <div className="candidate-vessel">

                    <Ship size={14} />

                    <span>
                      {nearest?.vessel_name ||
                        nearest?.name ||
                        "No AIS contact"}
                    </span>

                  </div>


                  <div className="candidate-distance">

                    {nearest?.distance_to_spill_km !=
                    null
                      ? `${Number(
                          nearest.distance_to_spill_km
                        ).toFixed(2)} km`
                      : "—"}

                  </div>


                  <div className="candidate-confidence">

                    <strong>
                      {Math.round(
                        confidence
                      )}
                      %
                    </strong>

                    <small>
                      U-Net confidence
                    </small>

                  </div>


                  <ChevronRight size={18} />

                </button>
              );
            }
          )}

        </div>
      )}

    </div>
  );
}


// ============================================================
// ENVIRONMENT PANEL
// ============================================================

function EnvironmentPanel({
  wind,
  current,
}) {
  const windSpeed =
    wind?.speed_knots ?? 8;

  const currentSpeed =
    current?.speed_knots ?? 1.2;

  const windFrom =
    wind?.wind_from_direction ??
    wind?.direction_degrees ??
    270;

  const windTo =
    wind?.wind_to_direction ??
    ((Number(windFrom) + 180) % 360);

  const currentDirection =
    current?.current_to_direction ??
    current?.direction_degrees ??
    90;

  return (
    <div className="panel environment-panel">

      <div className="panel-header">

        <div>
          <div className="eyebrow">
            ENVIRONMENT
          </div>

          <h2>
            Drift Conditions
          </h2>
        </div>

      </div>


      <div className="environment-content">

        <div className="environment-item">

          <div className="environment-icon">
            <Wind size={26} />
          </div>

          <div>

            <span>
              WIND
            </span>

            <strong>
              {windSpeed} kn
            </strong>

            <small>
              {windFrom}° FROM →{" "}
              {windTo}° TO
            </small>

          </div>

          <span className="condition-badge">
            MODERATE
          </span>

        </div>


        <div className="environment-item">

          <div className="environment-icon">
            <Waves size={26} />
          </div>

          <div>

            <span>
              OCEAN CURRENT
            </span>

            <strong>
              {currentSpeed} kn
            </strong>

            <small>
              {currentDirection}° TO direction
            </small>

          </div>

          <span className="condition-badge green">
            MODERATE
          </span>

        </div>

      </div>


      <div className="environment-note">

        <Navigation size={17} />

        Environmental alignment is a heuristic
        indicator and not a physical drift model.

      </div>

    </div>
  );
}


// ============================================================
// SELECTED CANDIDATE
// ============================================================

function SelectedCandidatePanel({
  candidate,
  wind,
  current,
  onClose,
}) {
  const nearest =
    getNearestVessel(candidate);

  const regionConfidenceRaw =
    Number(
      candidate?.sar_region_confidence ??
      candidate?.detector?.mean_confidence ??
      0
    );

  const regionConfidence =
    regionConfidenceRaw <= 1
      ? regionConfidenceRaw * 100
      : regionConfidenceRaw;

  const classifierConfidenceRaw =
    Number(candidate?.classifier_confidence ?? 0);

  const classifierConfidence =
    classifierConfidenceRaw <= 1
      ? classifierConfidenceRaw * 100
      : classifierConfidenceRaw;

  const windScore =
    getAlignmentScore(
      candidate.wind_analysis
    );

  const currentScore =
    getAlignmentScore(
      candidate.current_analysis
    );

  const attribution =
    nearest?.attribution?.priority_score ??
    nearest?.attribution?.score ??
    nearest?.attribution?.attribution_score ??
    nearest?.priority ??
    nearest?.correlation_score ??
    null;


  return (
    <div className="panel selected-panel">

      <div className="panel-header">

        <div>
          <div className="eyebrow">
            SELECTED DETECTION
          </div>

          <h2>
            Candidate{" "}
            {candidate.candidate_id}
          </h2>
        </div>

        <button
          type="button"
          className="icon-button"
          onClick={onClose}
        >
          <X size={19} />
        </button>

      </div>


      <div className="detail-grid">

        <Detail
          label="CLASSIFICATION"
          value={
            candidate.classification
          }
          subtitle={`Scene classifier ${classifierConfidence.toFixed(1)}%`}
          highlight
        />

        <Detail
          label="LOCATION"
          value={
            candidate.location
              ? `${candidate.location.latitude.toFixed(
                  6
                )}° N`
              : "Unavailable"
          }
          subtitle={
            candidate.location
              ? `${candidate.location.longitude.toFixed(
                  6
                )}° E`
              : ""
          }
        />

        <Detail
          label="U-NET REGION"
          value={`${regionConfidence.toFixed(1)}%`}
          subtitle="Segmentation confidence"
          highlight
        />

        <Detail
          label="SAR AREA"
          value={
            candidate.area_pixels != null
              ? `${Number(
                  candidate.area_pixels
                ).toLocaleString()} px`
              : "—"
          }
          subtitle={
            candidate.area_ratio != null
              ? `${(
                  Number(candidate.area_ratio) *
                  100
                ).toFixed(3)}% image area`
              : ""
          }
        />

        <Detail
          label="BACKSCATTER"
          value={
            candidate.mean_backscatter != null
              ? Number(
                  candidate.mean_backscatter
                ).toFixed(3)
              : "—"
          }
          subtitle={
            candidate.darkness_score != null
              ? `Darkness ${Number(
                  candidate.darkness_score
                ).toFixed(3)}`
              : ""
          }
        />

        <Detail
          label="GEOMETRY"
          value={
            candidate.dimensions
              ? `${candidate.dimensions.width_pixels} × ${candidate.dimensions.height_pixels}`
              : candidate.geometry?.bbox
                ? `${candidate.geometry.bbox.width} × ${candidate.geometry.bbox.height}`
                : "—"
          }
          subtitle={
            candidate.orientation != null
              ? `Orientation ${Number(
                  candidate.orientation
                ).toFixed(1)}°`
              : ""
          }
        />

        {(() => {
          const source = getEstimatedSource(candidate);
          const trajectory = getDriftTrajectory(candidate);
          const uncertainty = getSourceUncertainty(candidate);

          return (
            <>
              <Detail
                label="ESTIMATED SOURCE"
                value={
                  source
                    ? `${source.latitude.toFixed(5)}° N`
                    : "Unavailable"
                }
                subtitle={
                  source
                    ? `${source.longitude.toFixed(5)}° E`
                    : "Prototype backward drift"
                }
                highlight={Boolean(source)}
              />

              <Detail
                label="BACKWARD DRIFT"
                value={
                  trajectory.length > 1
                    ? `${trajectory.length - 1} h backtrack`
                    : getBackwardDrift(candidate)
                    ? "Available"
                    : "Unavailable"
                }
                subtitle={
                  uncertainty != null
                    ? `Source uncertainty ${uncertainty.toFixed(1)} km`
                    : "Prototype drift reconstruction"
                }
              />
            </>
          );
        })()}

        <Detail
          label="NEAREST AIS"
          value={
            nearest?.vessel_name ||
            nearest?.name ||
            "No vessel"
          }
          subtitle={
            nearest?.distance_to_spill_km !=
            null
              ? `${Number(
                  nearest.distance_to_spill_km
                ).toFixed(3)} km`
              : "Distance unavailable"
          }
        />

        <Detail
          label="ATTRIBUTION PRIORITY"
          value={
            attribution != null
              ? `${Math.round(
                  Number(attribution)
                )}/100`
              : "—"
          }
          subtitle="Heuristic prioritization — not causal proof"
        />

        <Detail
          label="WIND INPUT"
          value={
            wind?.speed_knots != null
              ? `${wind.speed_knots} kn`
              : "Prototype input"
          }
          subtitle={
            wind?.wind_from_direction != null
              ? `${wind.wind_from_direction}° FROM → ${wind.wind_to_direction ?? ((Number(wind.wind_from_direction) + 180) % 360)}° TO`
              : "Prototype environmental input"
          }
        />

        <Detail
          label="CURRENT INPUT"
          value={
            current?.speed_knots != null
              ? `${current.speed_knots} kn`
              : "Prototype input"
          }
          subtitle={
            current?.current_to_direction != null
              ? `${current.current_to_direction}° TO`
              : "Prototype environmental input"
          }
        />

      </div>


      <div className="disclaimer">

        <Navigation size={17} />

        SPILLTRACE confidence, AIS priority,
        and wind/current alignment are heuristic
        indicators. The backward-drift source is a
        prototype reconstruction using current
        development inputs; it does not establish
        causal source attribution.

      </div>

    </div>
  );
}


// ============================================================
// DETAIL
// ============================================================

function Detail({
  label,
  value,
  subtitle,
  highlight,
}) {
  return (
    <div className="detail-cell">

      <span>
        {label}
      </span>

      <strong
        className={
          highlight ? "highlight" : ""
        }
      >
        {value}
      </strong>

      <small>
        {subtitle}
      </small>

    </div>
  );
}


// ============================================================
// OVERVIEW PAGE
// ============================================================

function OverviewPage({
  candidates,
  vessels,
  wind,
  current,
  selectedCandidate,
  setSelectedCandidate,
  onAnalyze,
  analyzing,
  selectedFile,
  setSelectedFile,
}) {
  const reviewCount =
    candidates.filter(
      (candidate) => {
        const classification = String(
          candidate?.classification ||
          ""
        ).toUpperCase();

        return (
          classification === "REQUIRES REVIEW" ||
          classification === "LOOKALIKE" ||
          classification === "NO OIL" ||
          classification === "NO_OIL"
        );
      }
    ).length;

  return (
    <>
      <UploadPanel
        selectedFile={selectedFile}
        setSelectedFile={setSelectedFile}
        onAnalyze={onAnalyze}
        analyzing={analyzing}
      />

      <PageHeader />


      <div className="stats-grid">

        <StatCard
          icon={<CircleAlert size={22} />}
          label="DETECTIONS"
          value={String(
            candidates.length
          ).padStart(2, "0")}
          subtitle="SAR candidates"
        />

        <StatCard
          icon={<AlertTriangle size={22} />}
          label="REQUIRES REVIEW"
          value={String(
            reviewCount
          ).padStart(2, "0")}
          subtitle="No confirmed classification"
        />

        <StatCard
          icon={<Ship size={22} />}
          label="AIS CONTACTS"
          value={String(
            vessels.length
          ).padStart(2, "0")}
          subtitle="Development sample"
        />

        <StatCard
          icon={<Wind size={22} />}
          label="WIND"
          value={`${wind?.speed_knots ?? 8} kn`}
          subtitle={`${wind?.wind_from_direction ?? 270}° from / ${wind?.wind_to_direction ?? 90}° to`}
        />

      </div>


      <DetectionMap
        candidates={candidates}
        vessels={vessels}
        selectedCandidate={selectedCandidate}
        setSelectedCandidate={
          setSelectedCandidate
        }
      />


      <div className="content-grid">

        <CandidateList
          candidates={candidates}
          selectedCandidate={
            selectedCandidate
          }
          setSelectedCandidate={
            setSelectedCandidate
          }
        />

        <EnvironmentPanel
          wind={wind}
          current={current}
        />

      </div>


      {selectedCandidate && (
        <SelectedCandidatePanel
          candidate={selectedCandidate}
          wind={wind}
          current={current}
          onClose={() =>
            setSelectedCandidate(null)
          }
        />
      )}

    </>
  );
}


// ============================================================
// SAR PAGE
// ============================================================

function SARPage({
  candidates,
  selectedCandidate,
  wind,
  current,
  setSelectedCandidate,
  annotatedImage,
  selectedFile,
  setSelectedFile,
  onAnalyze,
  analyzing,
}) {
  return (
    <>
      <PageHeader />


      <UploadPanel
        selectedFile={selectedFile}
        setSelectedFile={setSelectedFile}
        onAnalyze={onAnalyze}
        analyzing={analyzing}
      />


      <CandidateList
        candidates={candidates}
        selectedCandidate={selectedCandidate}
        setSelectedCandidate={
          setSelectedCandidate
        }
      />


      {annotatedImage && (
        <div className="panel">

          <div className="panel-header">

            <div>
              <div className="eyebrow">
                SAR OUTPUT
              </div>

              <h2>
                Detection Analysis Image
              </h2>
            </div>

          </div>

          <div className="analysis-image-container">

            <img
              src={annotatedImage}
              alt="SAR detection analysis"
            />

          </div>

        </div>
      )}


      {selectedCandidate && (
        <SelectedCandidatePanel
          candidate={selectedCandidate}
          wind={wind}
          current={current}
          onClose={() =>
            setSelectedCandidate(null)
          }
        />
      )}

    </>
  );
}


// ============================================================
// AIS PAGE
// ============================================================

function AisPage({
  vessels,
}) {
  return (
    <>
      <PageHeader />

      <div className="stats-grid ais-stats">

        <StatCard
          icon={<Ship size={22} />}
          label="AIS CONTACTS"
          value={String(
            vessels.length
          ).padStart(2, "0")}
          subtitle="Available vessels"
        />

        <StatCard
          icon={<Navigation size={22} />}
          label="TRACKING"
          value="ACTIVE"
          subtitle="Development data"
        />

      </div>


      <div className="panel">

        <div className="panel-header">

          <div>
            <div className="eyebrow">
              AIS ATTRIBUTION
            </div>

            <h2>
              Vessel Contacts
            </h2>
          </div>

          <span className="map-tag">
            DEVELOPMENT DATA
          </span>

        </div>


        <div className="vessel-list">

          {vessels.map(
            (vessel, index) => (
              <div
                className="vessel-row"
                key={index}
              >

                <div className="vessel-icon">
                  <Ship size={21} />
                </div>

                <div className="vessel-name">

                  <strong>
                    {vessel.vessel_name ||
                      vessel.name ||
                      "Unknown Vessel"}
                  </strong>

                  <small>
                    {vessel.vessel_type ||
                      "VESSEL"}
                  </small>

                </div>

                <div className="vessel-data">

                  <span>
                    SPEED
                  </span>

                  <strong>
                    {vessel.speed_knots ??
                      "—"}{" "}
                    kn
                  </strong>

                </div>

                <div className="vessel-data">

                  <span>
                    HEADING
                  </span>

                  <strong>
                    {vessel.heading_degrees ??
                      vessel.heading ??
                      "—"}
                    °
                  </strong>

                </div>

                <div className="vessel-priority">

                  <strong>
                    {vessel.priority ??
                      vessel.attribution?.priority_score ??
                      0}
                  </strong>

                  <small>
                    PRIORITY
                  </small>

                </div>

              </div>
            )
          )}

        </div>

      </div>

    </>
  );
}


// ============================================================
// ENVIRONMENT PAGE
// ============================================================

function EnvironmentPage({
  wind,
  current,
  candidates,
}) {
  return (
    <>
      <PageHeader />

      <div className="stats-grid">

        <StatCard
          icon={<Wind size={22} />}
          label="WIND"
          value={`${wind?.speed_knots ?? 8} kn`}
          subtitle={`${wind?.wind_from_direction ?? 270}° FROM`}
        />

        <StatCard
          icon={<Waves size={22} />}
          label="OCEAN CURRENT"
          value={`${current?.speed_knots ?? 1.2} kn`}
          subtitle={`${current?.current_to_direction ?? 90}° TO`}
        />

        <StatCard
          icon={<Satellite size={22} />}
          label="SAR CANDIDATES"
          value={String(
            candidates.length
          ).padStart(2, "0")}
          subtitle="Environmental context"
        />

      </div>


      <div className="environment-large-grid">

        <div className="panel environment-large-card">

          <div className="panel-header">

            <div>
              <div className="eyebrow">
                ATMOSPHERIC DATA
              </div>

              <h2>
                Wind Conditions
              </h2>
            </div>

          </div>


          <div className="environment-big">

            <div className="environment-icon large">
              <Wind size={34} />
            </div>

            <div>

              <span>
                Wind
              </span>

              <strong>
                {wind?.speed_knots ?? 8} kn
              </strong>

              <small>
                {wind?.wind_from_direction ??
                  270}
                ° FROM →{" "}
                {wind?.wind_to_direction ??
                  90}
                ° TO
              </small>

              <small>
                Source:{" "}
                {wind?.source ||
                  wind?.data_source ||
                  "DEVELOPMENT_DATA"}
              </small>

            </div>

          </div>

        </div>


        <div className="panel environment-large-card">

          <div className="panel-header">

            <div>
              <div className="eyebrow">
                OCEANOGRAPHIC DATA
              </div>

              <h2>
                Ocean Current
              </h2>
            </div>

          </div>


          <div className="environment-big">

            <div className="environment-icon large">
              <Waves size={34} />
            </div>

            <div>

              <span>
                Ocean Current
              </span>

              <strong>
                {current?.speed_knots ?? 1.2} kn
              </strong>

              <small>
                {current?.current_to_direction ??
                  90}
                ° TO direction
              </small>

              <small>
                Source:{" "}
                {current?.source ||
                  current?.data_source ||
                  "DEVELOPMENT_DATA"}
              </small>

            </div>

          </div>

        </div>

      </div>


      <div className="panel limitation-panel">

        <div className="panel-header">

          <div>
            <div className="eyebrow">
              MODEL LIMITATION
            </div>

            <h2>
              Environmental Interpretation
            </h2>
          </div>

        </div>

        <div className="environment-note large-note">

          <Navigation size={20} />

          Wind/current alignment in SPILLTRACE
          is a heuristic indicator. It is not a
          physical oil-drift or ocean circulation
          model.

        </div>

      </div>

    </>
  );
}


// ============================================================
// HISTORY PAGE
// ============================================================

function HistoryPage({
  history,
  setSelectedCandidate,
}) {
  return (
    <>
      <PageHeader />

      <div className="panel">

        <div className="panel-header">

          <div>
            <div className="eyebrow">
              ANALYSIS
            </div>

            <h2>
              Detection History
            </h2>
          </div>

          <div className="count-badge">
            {String(history.length).padStart(
              2,
              "0"
            )}
          </div>

        </div>


        {history.length === 0 ? (
          <div className="empty-state">

            <History size={42} />

            <h3>
              No analysis history
            </h3>

            <p>
              Your completed SAR analyses
              will appear here.
            </p>

          </div>
        ) : (
          <div className="history-list">

            {history.map(
              (item, index) => (
                <div
                  className="history-row"
                  key={
                    item.spill_id ||
                    index
                  }
                >

                  <div>
                    <strong>
                      {item.spill_id ||
                        "Analysis"}
                    </strong>

                    <small>
                      {item.timestamp
                        ? new Date(
                            item.timestamp
                          ).toLocaleString()
                        : "Unknown time"}
                    </small>
                  </div>

                  <div>
                    <strong>
                      {item.candidate_count ??
                        item.candidates
                          ?.length ??
                        0}
                    </strong>

                    <small>
                      candidates
                    </small>
                  </div>

                  <button
                    type="button"
                    className="small-button"
                    onClick={() => {
                      const candidate =
                        item.candidates?.[0];

                      if (candidate) {
                        setSelectedCandidate(
                          candidate
                        );
                      }
                    }}
                  >
                    View
                  </button>

                </div>
              )
            )}

          </div>
        )}

      </div>

    </>
  );
}


// ============================================================
// DATA SOURCES
// ============================================================

function DataSourcesPage() {
  const sources = [
    {
      name: "Sentinel SAR",
      type: "Satellite imagery",
      status: "CONNECTED",
      icon: <Satellite size={24} />,
    },
    {
      name: "AIS",
      type: "Vessel tracking",
      status: "DEVELOPMENT DATA",
      icon: <Ship size={24} />,
    },
    {
      name: "Wind",
      type: "Atmospheric conditions",
      status: "DEVELOPMENT DATA",
      icon: <Wind size={24} />,
    },
    {
      name: "Ocean Current",
      type: "Oceanographic conditions",
      status: "DEVELOPMENT DATA",
      icon: <Waves size={24} />,
    },
  ];

  return (
    <>
      <PageHeader />

      <div className="panel">

        <div className="panel-header">

          <div>
            <div className="eyebrow">
              DATA SOURCES
            </div>

            <h2>
              Intelligence Inputs
            </h2>
          </div>

        </div>


        <div className="source-grid">

          {sources.map(
            (source) => (
              <div
                className="source-card"
                key={source.name}
              >

                <div className="source-icon">
                  {source.icon}
                </div>

                <div>

                  <strong>
                    {source.name}
                  </strong>

                  <small>
                    {source.type}
                  </small>

                  <span>
                    {source.status}
                  </span>

                </div>

              </div>
            )
          )}

        </div>

      </div>

    </>
  );
}


// ============================================================
// SYSTEM STATUS
// ============================================================

function SystemStatusPage({
  backendOnline,
}) {
  return (
    <>
      <PageHeader />

      <div className="panel">

        <div className="panel-header">

          <div>
            <div className="eyebrow">
              SYSTEM
            </div>

            <h2>
              System Status
            </h2>
          </div>

        </div>


        <div className="status-list">

          <StatusRow
            label="Frontend"
            status="ONLINE"
            ok
          />

          <StatusRow
            label="FastAPI Backend"
            status={
              backendOnline
                ? "ONLINE"
                : "OFFLINE"
            }
            ok={backendOnline}
          />

          <StatusRow
            label="SAR Detection Engine"
            status={
              backendOnline
                ? "READY"
                : "WAITING"
            }
            ok={backendOnline}
          />

          <StatusRow
            label="AIS Attribution"
            status="DEVELOPMENT DATA"
            ok
          />

          <StatusRow
            label="Environment Analysis"
            status="DEVELOPMENT DATA"
            ok
          />

        </div>

      </div>

    </>
  );
}


function StatusRow({
  label,
  status,
  ok,
}) {
  return (
    <div className="status-row">

      <div>

        {ok ? (
          <CheckCircle2 size={18} />
        ) : (
          <WifiOff size={18} />
        )}

        <strong>
          {label}
        </strong>

      </div>

      <span
        className={
          ok
            ? "status-online"
            : "status-offline"
        }
      >
        {status}
      </span>

    </div>
  );
}


// ============================================================
// APP
// ============================================================

export default function App() {

  const [activePage, setActivePage] =
    useState("overview");

  const [backendOnline, setBackendOnline] =
    useState(false);

  const [selectedFile, setSelectedFile] =
    useState(null);

  const [analyzing, setAnalyzing] =
    useState(false);

  const [candidates, setCandidates] =
    useState([]);

  const [hasAnalyzed, setHasAnalyzed] =
    useState(false);

  const [detectionResult, setDetectionResult] =
    useState(null);

  const [vessels, setVessels] =
    useState(DEMO_VESSELS);

  const [selectedCandidate, setSelectedCandidate] =
    useState(null);

  const [annotatedImage, setAnnotatedImage] =
    useState(null);

  const [wind, setWind] =
    useState({
      speed_knots: 8,
      wind_from_direction: 270,
      wind_to_direction: 90,
      source: "DEVELOPMENT_DATA",
    });

  const [current, setCurrent] =
    useState({
      speed_knots: 1.2,
      current_to_direction: 90,
      source: "DEVELOPMENT_DATA",
    });

  const [history, setHistory] =
    useState([]);


  // ==========================================================
  // LOAD SAVED DATA
  // ==========================================================

  useEffect(() => {
    try {
      const saved =
        localStorage.getItem(
          "spilltrace_history"
        );

      if (saved) {
        const parsed =
          JSON.parse(saved);

        if (Array.isArray(parsed)) {
          setHistory(parsed);
        }
      }


      const savedLatest =
        localStorage.getItem(
          "spilltrace_latest"
        );

      if (savedLatest) {
        const latest =
          JSON.parse(savedLatest);

        if (latest.detectionResult) {
          setDetectionResult(latest.detectionResult);
          setHasAnalyzed(true);
        }

        if (
          Array.isArray(
            latest.candidates
          ) &&
          latest.candidates.length > 0
        ) {
          setCandidates(
            latest.candidates.map(
              normalizeCandidate
            )
          );

          if (
            Array.isArray(
              latest.vessels
            )
          ) {
            setVessels(
              latest.vessels
            );
          }

          if (latest.wind) {
            setWind(latest.wind);
          }

          if (latest.current) {
            setCurrent(
              latest.current
            );
          }

          if (
            latest.annotatedImage
          ) {
            setAnnotatedImage(
              latest.annotatedImage
            );
          }
        }
      }

    } catch (error) {
      console.error(
        "Could not load saved data:",
        error
      );
    }
  }, []);


  // ==========================================================
  // BACKEND HEALTH CHECK
  // ==========================================================

  const checkBackend =
    async () => {
      try {
        const response =
          await fetch(
            `${API_URL}/health`,
            {
              method: "GET",
            }
          );

        if (!response.ok) {
          throw new Error(
            "Backend unavailable"
          );
        }

        setBackendOnline(true);

      } catch {
        setBackendOnline(false);
      }
    };


  useEffect(() => {
    checkBackend();

    const interval =
      setInterval(
        checkBackend,
        10000
      );

    return () =>
      clearInterval(interval);
  }, []);


  // ==========================================================
  // ANALYZE IMAGE
  // ==========================================================

  const analyzeImage =
    async () => {
      if (!selectedFile) {
        return;
      }

      setAnalyzing(true);
      setSelectedCandidate(null);

      try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(
          `${API_URL}/detect-spill`,
          {
            method: "POST",
            body: formData,
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data?.error ||
            data?.detail ||
            `HTTP ${response.status}`
          );
        }

        if (data?.success === false) {
          throw new Error(
            data?.error ||
            data?.detail ||
            "Analysis failed"
          );
        }

        setHasAnalyzed(true);
        setDetectionResult(data);

        const sceneClass =
          data?.classification?.class ||
          data?.classification?.label ||
          "UNKNOWN";

        const sceneConfidence = Number(
          data?.classification?.confidence ?? 0
        );

        const sceneProbabilities =
          data?.classification?.probabilities || {};

        // Accept the multi-region response under any of
        // the common keys used by the backend.
        let apiCandidates = [];

        if (Array.isArray(data?.candidates)) {
          apiCandidates = data.candidates;
        } else if (Array.isArray(data?.detections)) {
          apiCandidates = data.detections;
        } else if (Array.isArray(data?.regions)) {
          apiCandidates = data.regions;
        }

        // Backward compatibility with the original
        // single-region response.
        if (
          apiCandidates.length === 0 &&
          data?.spill_detected &&
          (
            data?.location ||
            data?.geometry ||
            data?.geographic_location
          )
        ) {
          apiCandidates = [
            {
              candidate_id:
                data?.spill_id || "SPILL-01",

              location:
                data?.location ||
                data?.geographic_location ||
                null,

              geometry:
                data?.geometry || null,

              bbox:
                data?.geometry?.bbox || null,

              centroid:
                data?.geometry?.centroid || null,

              area_pixels:
                data?.geometry?.area_pixels || 0,

              confidence:
                data?.detector?.mean_confidence ?? 0,
            },
          ];
        }

        const normalized =
          apiCandidates
            .map((candidate, index) => {
              if (!candidate) {
                return null;
              }

              const rawLocation =
                candidate?.location ||
                candidate?.geographic_location ||
                (
                  candidate?.latitude != null &&
                  candidate?.longitude != null
                    ? {
                        latitude: candidate.latitude,
                        longitude: candidate.longitude,
                      }
                    : null
                );

              const location =
                rawLocation &&
                Number.isFinite(
                  Number(rawLocation.latitude)
                ) &&
                Number.isFinite(
                  Number(rawLocation.longitude)
                )
                  ? {
                      latitude:
                        Number(rawLocation.latitude),
                      longitude:
                        Number(rawLocation.longitude),
                    }
                  : null;

              const bbox =
                candidate?.bbox ||
                candidate?.geometry?.bbox ||
                null;

              const regionConfidence = Number(
                candidate?.confidence ??
                candidate?.mean_confidence ??
                candidate?.region_confidence ??
                candidate?.oil_confidence ??
                0
              );

              return normalizeCandidate({
                ...candidate,

                candidate_id:
                  candidate?.candidate_id ||
                  candidate?.id ||
                  `${data?.spill_id || "SPILL"}-${String(
                    index + 1
                  ).padStart(2, "0")}`,

                classification:
                  sceneClass,

                classifier:
                  data?.classification?.model ||
                  "spilltrace_classifier.pth",

                classifier_confidence:
                  sceneConfidence,

                class_probabilities:
                  sceneProbabilities,

                model:
                  data?.classification?.model ||
                  null,

                oil_confidence:
                  regionConfidence,

                validation_score:
                  regionConfidence <= 1
                    ? regionConfidence * 100
                    : regionConfidence,

                spill_detected: true,

                location,

                area_pixels: Number(
                  candidate?.area_pixels ??
                  candidate?.geometry?.area_pixels ??
                  0
                ),

                dimensions:
                  bbox
                    ? {
                        width_pixels:
                          Number(bbox.width ?? 0),
                        height_pixels:
                          Number(bbox.height ?? 0),
                      }
                    : null,

                geometry:
                  candidate?.geometry ||
                  {
                    bbox,
                    centroid:
                      candidate?.centroid ||
                      null,
                    area_pixels:
                      candidate?.area_pixels ?? 0,
                  },

                spill_id:
                  data?.spill_id || null,

                ais_vessels:
                  Array.isArray(candidate?.ais_vessels)
                    ? candidate.ais_vessels
                    : [],

                wind_analysis:
                  candidate?.wind_analysis ||
                  {
                    status:
                      candidate?.wind_status ||
                      "NOT_CONNECTED",
                  },

                current_analysis:
                  candidate?.current_analysis ||
                  {
                    status:
                      candidate?.current_status ||
                      "NOT_CONNECTED",
                  },
              });
            })
            .filter(Boolean);

        // A real analysis must never fall back to demo detections.
        // AIS can be returned either at top-level or inside each
        // detection candidate. Collect both forms for the UI.
        const realVessels =
          collectAisVessels(data, normalized);

        // Keep the normalized AIS list on each candidate too.
        const candidatesWithAis =
          normalized.map((candidate, index) => {
            // Backward-drift compatibility: some backend versions
            // return the drift/source at top level while others
            // return it inside each candidate. Preserve both forms.
            const topDrift =
              index === 0
                ? (data?.backward_drift || null)
                : null;

            const backwardDrift =
              candidate?.backward_drift ||
              topDrift ||
              null;

            const topSource =
              index === 0
                ? (data?.estimated_source || null)
                : null;

            const estimatedSource =
              candidate?.estimated_source ||
              topSource ||
              backwardDrift?.estimated_source ||
              backwardDrift?.source ||
              backwardDrift?.origin ||
              (Array.isArray(backwardDrift?.trajectory) &&
              backwardDrift.trajectory.length > 0
                ? backwardDrift.trajectory[
                    backwardDrift.trajectory.length - 1
                  ]
                : null);

            const topUncertainty =
              index === 0
                ? data?.source_uncertainty_km
                : null;

            const sourceUncertainty =
              candidate?.source_uncertainty_km ??
              topUncertainty ??
              backwardDrift?.uncertainty_radius_km ??
              backwardDrift?.source_uncertainty_km ??
              null;

            return {
              ...candidate,
              backward_drift: backwardDrift,
              estimated_source: estimatedSource,
              source_uncertainty_km: sourceUncertainty,
              ais_vessels:
                getCandidateVessels(candidate),
            };
          });

        setCandidates(candidatesWithAis);
        setVessels(realVessels);

        const backendWind =
          data?.wind ||
          data?.environment?.wind ||
          data?.wind_analysis ||
          null;

        const backendCurrent =
          data?.current ||
          data?.environment?.current ||
          data?.ocean_current ||
          data?.current_analysis ||
          null;

        setWind(
          backendWind
            ? {
                ...backendWind,
                source:
                  backendWind.source ||
                  backendWind.data_source ||
                  "SYNTHETIC PROTOTYPE",
              }
            : {
                source: "NOT_CONNECTED",
                status: "NOT_CONNECTED",
                speed_knots: null,
                wind_from_direction: null,
                wind_to_direction: null,
              }
        );

        setCurrent(
          backendCurrent
            ? {
                ...backendCurrent,
                source:
                  backendCurrent.source ||
                  backendCurrent.data_source ||
                  "SYNTHETIC PROTOTYPE",
              }
            : {
                source: "NOT_CONNECTED",
                status: "NOT_CONNECTED",
                speed_knots: null,
                current_to_direction: null,
              }
        );

        let imageUrl = null;

        if (data?.prediction_mask?.url) {
          imageUrl =
            `${API_URL}${data.prediction_mask.url}`;

          setAnnotatedImage(imageUrl);
        } else {
          setAnnotatedImage(null);
        }

        const historyEntry = {
          spill_id:
            data?.spill_id ||
            `SP-${Date.now()}`,

          timestamp:
            data?.timestamp ||
            new Date().toISOString(),

          candidate_count:
            candidatesWithAis.length,

          classification:
            sceneClass,

          classifier_confidence:
            sceneConfidence,

          candidates:
            candidatesWithAis,

          vessels: realVessels,

          wind: {
            ...(backendWind || {}),
            source:
              backendWind?.source ||
              backendWind?.data_source ||
              "PROTOTYPE INPUT",
          },

          current: {
            ...(backendCurrent || {}),
            source:
              backendCurrent?.source ||
              backendCurrent?.data_source ||
              "PROTOTYPE INPUT",
          },

          annotatedImage:
            imageUrl,

          detectionResult:
            data,
        };

        const updatedHistory = [
          historyEntry,
          ...history,
        ].slice(0, 20);

        setHistory(updatedHistory);

        localStorage.setItem(
          "spilltrace_history",
          JSON.stringify(updatedHistory)
        );

        localStorage.setItem(
          "spilltrace_latest",
          JSON.stringify({
            hasAnalyzed: true,
            detectionResult: data,
            candidates: candidatesWithAis,
            vessels: realVessels,
            wind: historyEntry.wind,
            current: historyEntry.current,
            annotatedImage: imageUrl,
          })
        );

        if (candidatesWithAis.length > 0) {
          setSelectedCandidate(
            candidatesWithAis[0]
          );
        }

        setActivePage("overview");

      } catch (error) {
        console.error(
          "SPILLTRACE analysis error:",
          error
        );

        alert(
          `Analysis failed:\n\n${
            error?.message || error
          }\n\nMake sure the FastAPI backend is running on ${API_URL}.`
        );
      } finally {
        setAnalyzing(false);
      }
    };


  // ==========================================================
  // DISPLAY DATA
  // ==========================================================

  const displayCandidates =
    hasAnalyzed
      ? candidates
      : DEMO_CANDIDATES;


  const displayVessels =
    hasAnalyzed
      ? vessels
      : DEMO_VESSELS;


  // ==========================================================
  // PAGE ROUTING
  // ==========================================================

  const page = useMemo(() => {

    switch (activePage) {

      case "sar":
        return (
          <SARPage
            candidates={
              displayCandidates
            }
            selectedCandidate={
              selectedCandidate
            }
            wind={wind}
            current={current}
            setSelectedCandidate={
              setSelectedCandidate
            }
            annotatedImage={
              annotatedImage
            }
            selectedFile={
              selectedFile
            }
            setSelectedFile={
              setSelectedFile
            }
            onAnalyze={
              analyzeImage
            }
            analyzing={
              analyzing
            }
          />
        );


      case "ais":
        return (
          <AisPage
            vessels={
              displayVessels
            }
          />
        );


      case "environment":
        return (
          <EnvironmentPage
            wind={wind}
            current={current}
            candidates={
              displayCandidates
            }
          />
        );


      case "history":
        return (
          <HistoryPage
            history={history}
            setSelectedCandidate={
              setSelectedCandidate
            }
          />
        );


      case "sources":
        return (
          <DataSourcesPage />
        );


      case "status":
        return (
          <SystemStatusPage
            backendOnline={
              backendOnline
            }
          />
        );


      case "overview":
      default:
        return (
          <OverviewPage
            candidates={
              displayCandidates
            }
            vessels={
              displayVessels
            }
            wind={wind}
            current={current}
            selectedCandidate={
              selectedCandidate
            }
            setSelectedCandidate={
              setSelectedCandidate
            }
            onAnalyze={
              analyzeImage
            }
            analyzing={
              analyzing
            }
            selectedFile={
              selectedFile
            }
            setSelectedFile={
              setSelectedFile
            }
          />
        );
    }

  }, [
    activePage,
    displayCandidates,
    displayVessels,
    selectedCandidate,
    annotatedImage,
    selectedFile,
    analyzing,
    wind,
    current,
    history,
    backendOnline,
    hasAnalyzed,
    detectionResult,
  ]);


  // ==========================================================
  // APP LAYOUT
  // ==========================================================

  return (
    <div className="app">

      <Sidebar
        activePage={
          activePage
        }
        setActivePage={
          setActivePage
        }
        backendOnline={
          backendOnline
        }
      />


      <main className="main-content">

        {page}

      </main>

    </div>
  );
}