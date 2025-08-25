
#!/usr/bin/env python3

import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math
import os
import time
import tempfile
import unicodedata
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import io
import base64

warnings.filterwarnings('ignore')


# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="ByteBrake AI - Brake System Engineering Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SPEED PATTERN KNOWLEDGE BASE - HARDCODED FOR PERFORMANCE
SPEED_KNOWLEDGE_BASE = {
    'city_traffic': {
        'base_speed': 25,
        'speed_variation': 15,
        'pattern': [0.3, 0.6, 0.8, 1.0, 0.7, 0.4, 0.2, 0.5, 0.9, 0.6, 0.4, 0.7, 0.9, 0.5, 0.3]
    },
    'highway': {
        'base_speed': 80,
        'speed_variation': 20,
        'pattern': [0.7, 0.8, 0.9, 1.0, 0.95, 0.85, 0.9, 1.0, 0.8, 0.7, 0.75, 0.85, 0.95, 1.0, 0.9]
    },
    'mixed': {
        'base_speed': 45,
        'speed_variation': 25,
        'pattern': [0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.6, 0.8, 0.95, 1.0, 0.85, 0.7, 0.6, 0.8, 0.9, 0.75, 0.5]
    },
    'aggressive': {
        'base_speed': 60,
        'speed_variation': 35,
        'pattern': [0.4, 0.7, 0.9, 1.0, 0.6, 0.3, 0.8, 1.0, 0.5, 0.2, 0.9, 1.0, 0.7, 0.4, 0.8, 0.95, 0.6, 0.9, 1.0, 0.3]
    }
}

# Enhanced constants with seasonal weather


class BrakeConstants:
    # Physical constants
    C_P = 470  # Specific heat capacity (J/kg·K)
    DENSITY = 2500  # Brake pad density (kg/m³)
    MU = 0.4  # Friction coefficient
    SIGMA = 5.67e-8  # Stefan-Boltzmann constant
    EPSILON = 0.71  # Emissivity for brake materials
    TIME_STEPS = 20  # Number of simulation time steps
    RUL_CORRECTION_FACTOR = 1.4  # Updated correction factor (hidden from UI)

    # Updated RUL upper bound
    MAX_RUL_KM = 30000  # Maximum RUL set to 30,000 km

    # Seasonal ambient temperatures (Celsius) - Enhanced for Indian conditions
    AMBIENT_TEMPS = {
        'summer': 45,  # Hot Indian summer
        'winter': 15,  # Indian winter
        'rainy': 28,   # Monsoon season
        'autumn': 32   # Post-monsoon
    }

    # Material temperature limits and properties
    MATERIAL_LIMITS = {
        'organic': {'max_temp': 280, 'wear_rate': 0.15, 'friction_coeff': 0.35},
        'sintered': {'max_temp': 450, 'wear_rate': 0.10, 'friction_coeff': 0.45},
        'ceramic': {'max_temp': 650, 'wear_rate': 0.07, 'friction_coeff': 0.55}
    }

    # Enhanced weather factors for Indian conditions
    WEATHER_FACTORS = {
        'summer': {'temp_mult': 1.2, 'wear_mult': 1.1, 'cooling_eff': 0.8},
        'winter': {'temp_mult': 0.7, 'wear_mult': 0.9, 'cooling_eff': 1.3},
        'rainy': {'temp_mult': 0.8, 'wear_mult': 1.4, 'cooling_eff': 1.5},
        'autumn': {'temp_mult': 1.0, 'wear_mult': 1.0, 'cooling_eff': 1.0}
    }

# ISO and Indian Standards Compliance


class ComplianceStandards:
    ISO_STANDARDS = {
        'min_rotor_diameter_mm': 160,
        'max_rotor_diameter_mm': 400,
        'min_pad_thickness_mm': 4.0,
        'max_pad_thickness_mm': 20.0,
        'min_friction_coefficient': 0.35,
        'max_friction_coefficient': 0.65,
        'max_operating_temp_c': 650
    }

    INDIAN_STANDARDS = {
        'ais_043_compliance': True,  # AIS-043 (Automotive Industry Standard)
        'bis_compliance': True,  # Bureau of Indian Standards
        'min_stopping_distance_factor': 1.0,
        'dust_resistance_required': True,
        'monsoon_performance_required': True
    }

# NEW: Speed Pattern Generator Function (existing)


def generate_speed_pattern_for_distance(distance_km: int, riding_style: str = 'mixed') -> Tuple[List[float], List[float]]:
    """
    Generate speed pattern for a given distance range based on riding style
    Returns: (distance_points, speed_points)
    """
    if riding_style not in SPEED_KNOWLEDGE_BASE:
        riding_style = 'mixed'  # Default fallback

    pattern_data = SPEED_KNOWLEDGE_BASE[riding_style]
    base_speed = pattern_data['base_speed']
    speed_variation = pattern_data['speed_variation']
    pattern = pattern_data['pattern']

    # Generate distance points
    # Between 50-500 points based on distance
    num_points = min(max(50, distance_km // 200), 500)
    distance_points = np.linspace(0, distance_km, num_points)

    # Generate speed pattern
    speed_points = []
    pattern_length = len(pattern)

    for i, dist in enumerate(distance_points):
        # Use pattern cyclically
        pattern_index = i % pattern_length
        pattern_multiplier = pattern[pattern_index]

        # Add some randomness for realism
        noise = np.random.uniform(-0.1, 0.1)
        speed_multiplier = max(0.1, pattern_multiplier + noise)

        # Calculate speed
        speed = base_speed + (speed_variation * (speed_multiplier - 0.5))
        speed = max(5, speed)  # Minimum 5 km/h
        speed_points.append(speed)

    return distance_points.tolist(), speed_points

# NEW: Temperature Calculation Based on Speed (existing)


def calculate_temperature_from_speed(speed_kmh: float, brake_params: Dict[str, Any],
                                     ambient_temp: float = 35) -> float:
    """
    Calculate brake temperature rise based on speed
    """
    speed_ms = speed_kmh / 3.6

    # Kinetic energy calculation
    vehicle_mass = brake_params.get('vehicle_mass_kg', 150)  # Default 150kg
    kinetic_energy = 0.5 * vehicle_mass * speed_ms**2

    # Heat generation (simplified model)
    pad_mass = brake_params.get('pad_mass_kg', 0.5)
    if pad_mass > 0:
        temp_rise = kinetic_energy / \
            (pad_mass * BrakeConstants.C_P * 10)  # Simplified
    else:
        temp_rise = speed_kmh * 0.8  # Fallback calculation

    # Temperature with ambient
    final_temp = ambient_temp + temp_rise

    # Add speed-dependent factor
    speed_factor = 1 + (speed_kmh / 100) * 0.5
    final_temp *= speed_factor

    return min(final_temp, 400)  # Cap at 400°C for realism

# NEW: Create Speed vs Distance Chart with Temperature (existing)


def create_speed_distance_temperature_chart(distance_km: int, riding_style: str,
                                            brake_params: Dict[str, Any]) -> go.Figure:
    """
    Create combined chart showing speed vs distance and corresponding temperature
    """
    # Generate speed pattern
    distances, speeds = generate_speed_pattern_for_distance(
        distance_km, riding_style)

    # Calculate temperatures for each speed point
    ambient_temp = BrakeConstants.AMBIENT_TEMPS['summer']  # Default to summer
    temperatures = [calculate_temperature_from_speed(speed, brake_params, ambient_temp)
                    for speed in speeds]

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Speed trace
    fig.add_trace(
        go.Scatter(x=distances, y=speeds, name="Speed (km/h)",
                   line=dict(color='#1f77b4', width=2),
                   mode='lines'),
        secondary_y=False,
    )

    # Temperature trace
    fig.add_trace(
        go.Scatter(x=distances, y=temperatures, name="Temperature (°C)",
                   line=dict(color='#ff7f0e', width=2),
                   mode='lines'),
        secondary_y=True,
    )

    # Update layout
    fig.update_xaxes(title_text="Distance (km)")
    fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
    fig.update_yaxes(title_text="Brake Temperature (°C)", secondary_y=True)

    fig.update_layout(
        title=f"Speed vs Distance Pattern with Brake Temperature ({riding_style.title()} Style)",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    return fig

# NEW: Create Temperature vs Distance Chart (NEW ADDITION)


def create_temperature_distance_chart(distance_km: int, riding_style: str,
                                      brake_params: Dict[str, Any]) -> go.Figure:
    """
    Create dedicated Temperature vs Distance chart based on speed patterns
    """
    # Generate speed pattern
    distances, speeds = generate_speed_pattern_for_distance(
        distance_km, riding_style)

    # Calculate temperatures for different weather conditions
    weather_temps = {}
    for weather in ['summer', 'winter', 'rainy', 'autumn']:
        ambient_temp = BrakeConstants.AMBIENT_TEMPS[weather]
        temperatures = [calculate_temperature_from_speed(speed, brake_params, ambient_temp)
                        for speed in speeds]
        weather_temps[weather] = temperatures

    # Create the chart
    fig = go.Figure()

    # Colors for different seasons
    colors = {
        'summer': '#FF4444',    # Red
        'winter': '#4444FF',    # Blue
        'rainy': '#44FF44',     # Green
        'autumn': '#FF8844'     # Orange
    }

    # Add traces for each weather condition
    for weather, temps in weather_temps.items():
        fig.add_trace(go.Scatter(
            x=distances,
            y=temps,
            mode='lines+markers',
            name=f'{weather.title()} Season',
            line=dict(color=colors[weather], width=3),
            marker=dict(size=4),
            hovertemplate=f"<b>{weather.title()} Season</b><br>" +
            "Distance: %{x:.1f} km<br>" +
                         "Temperature: %{y:.1f}°C<extra></extra>"
        ))

    fig.update_layout(
        title=f"Brake Temperature vs Distance ({riding_style.title()} Riding Style)",
        xaxis_title="Distance (km)",
        yaxis_title="Brake Temperature (°C)",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    return fig

# Utility functions (existing)


def clean_for_display(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace('_', ' ').title()
    replacements = {
        'Kg': 'kg', 'Kmh': 'km/h', 'Km': 'km', 'Cc': 'CC', 'Hp': 'HP',
        'Mm': 'mm', 'M2': 'm²', 'Deg': '°'
    }

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def clean_text_for_pdf(text: str) -> str:
    if not text:
        return ""
    replacements = {
        '\u2022': '* ', '\u2013': '-', '\u2014': '-', '\u2018': "'",
        '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u00b0': ' deg'
    }

    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    try:
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('latin-1', 'ignore').decode('latin-1')
    except:
        text = ''.join(char for char in text if ord(char) < 128)
    return text

# Brake calculations (existing)


def choose_pad_type(cc: float, hp: float) -> str:
    if cc < 200:
        return "organic"
    elif 200 <= cc <= 400:
        return "sintered"
    elif cc > 400 or hp > 25:
        return "ceramic"
    else:
        return "organic"


def get_stopping_time(max_speed_kmh: float) -> float:
    if max_speed_kmh <= 50:
        return 2.0
    elif max_speed_kmh <= 100:
        return 3.5
    elif max_speed_kmh <= 150:
        return 5.0
    else:
        return 6.0


def calculate_brake_parameters(bike_specs: Dict[str, Any], custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    # Basic conversions
    wheel_diameter_m = bike_specs['wheel_diameter_inch'] * 0.0254
    total_mass_kg = bike_specs['bike_weight_kg']
    max_speed_ms = bike_specs['max_speed_kmh'] / 3.6

    # Material selection (can be overridden by custom params)
    pad_type = choose_pad_type(
        bike_specs['engine_cc'], bike_specs['horsepower'])
    if custom_params and 'pad_type' in custom_params:
        pad_type = custom_params['pad_type']

    # Brake system calculations
    rotor_diameter_mm = wheel_diameter_m * 150
    if custom_params and 'rotor_diameter_mm' in custom_params:
        rotor_diameter_mm = custom_params['rotor_diameter_mm']
    else:
        rotor_diameter_mm = max(160, min(rotor_diameter_mm, 320))

    # Test velocity at 90% of max speed
    test_velocity_ms = max_speed_ms * 0.9

    # Kinetic energy at test speed
    kinetic_energy = 0.5 * total_mass_kg * test_velocity_ms ** 2

    # Stopping time based on speed ranges
    stopping_time = get_stopping_time(bike_specs['max_speed_kmh'])

    # Brake force calculation
    deceleration = test_velocity_ms / stopping_time
    brake_force_n = total_mass_kg * deceleration / 2  # Per wheel

    # Pad calculations (can be customized)
    pad_area_mm2 = rotor_diameter_mm * 25
    pad_thickness_mm = custom_params.get(
        'pad_thickness_mm', 10.0) if custom_params else 10.0
    pad_mass_kg = (pad_area_mm2 * pad_thickness_mm *
                   BrakeConstants.DENSITY) / 1e9

    # Radiative area calculation
    rotor_thickness_m = 0.012
    radiative_area_m2 = math.pi * \
        (rotor_diameter_mm/1000) * rotor_thickness_m * 2

    # Heat calculations
    max_temp_rise = kinetic_energy / (pad_mass_kg * BrakeConstants.C_P)

    # Stopping distance
    mu = custom_params.get(
        'friction_coefficient', BrakeConstants.MU) if custom_params else BrakeConstants.MU
    stopping_distance_m = (test_velocity_ms ** 2) / (2 * mu * 9.81)

    return {
        'pad_type': pad_type,
        'rotor_diameter_mm': round(rotor_diameter_mm, 1),
        'pad_thickness_mm': round(pad_thickness_mm, 2),
        'pad_area_mm2': round(pad_area_mm2, 1),
        'pad_mass_kg': round(pad_mass_kg, 4),
        'radiative_area_m2': round(radiative_area_m2, 6),
        'brake_force_n': round(brake_force_n, 1),
        'stopping_distance_m': round(stopping_distance_m, 1),
        'max_temp_rise_c': round(max_temp_rise, 1),
        'brake_power_w': round(brake_force_n * test_velocity_ms, 1),
        'kinetic_energy_j': round(kinetic_energy, 1),
        'test_velocity_ms': round(test_velocity_ms, 1),
        'stopping_time_s': stopping_time,
        'friction_coefficient': mu,
        'vehicle_mass_kg': total_mass_kg  # Store for temperature calculations
    }

# Compliance Validation Agent (existing)


def validate_brake_compliance(brake_params: Dict[str, Any]) -> Dict[str, Any]:
    iso_checks = {}
    indian_checks = {}
    recommendations = []

    # ISO Standard Checks
    rotor_diameter = brake_params.get('rotor_diameter_mm', 0)
    pad_thickness = brake_params.get('pad_thickness_mm', 0)
    pad_type = brake_params.get('pad_type', 'unknown')

    # Check rotor diameter
    iso_checks['rotor_diameter_compliant'] = (
        ComplianceStandards.ISO_STANDARDS['min_rotor_diameter_mm'] <= rotor_diameter
        <= ComplianceStandards.ISO_STANDARDS['max_rotor_diameter_mm']
    )

    # Check pad thickness
    iso_checks['pad_thickness_compliant'] = (
        ComplianceStandards.ISO_STANDARDS['min_pad_thickness_mm'] <= pad_thickness
        <= ComplianceStandards.ISO_STANDARDS['max_pad_thickness_mm']
    )

    # Check friction coefficient
    material_friction = BrakeConstants.MATERIAL_LIMITS.get(
        pad_type, {}).get('friction_coeff', 0.4)
    iso_checks['friction_coefficient_compliant'] = (
        ComplianceStandards.ISO_STANDARDS['min_friction_coefficient'] <= material_friction
        <= ComplianceStandards.ISO_STANDARDS['max_friction_coefficient']
    )

    # Indian Standard Checks
    indian_checks['ais_043_compliant'] = iso_checks['rotor_diameter_compliant'] and iso_checks['pad_thickness_compliant']
    indian_checks['bis_compliant'] = material_friction >= 0.35
    indian_checks['dust_resistant'] = pad_type in ['sintered', 'ceramic']
    indian_checks['monsoon_ready'] = pad_type == 'sintered' or pad_type == 'ceramic'

    # Generate recommendations
    if not iso_checks['rotor_diameter_compliant']:
        if rotor_diameter < ComplianceStandards.ISO_STANDARDS['min_rotor_diameter_mm']:
            recommendations.append(
                f"Increase rotor diameter to minimum {ComplianceStandards.ISO_STANDARDS['min_rotor_diameter_mm']}mm")
        else:
            recommendations.append(
                f"Reduce rotor diameter to maximum {ComplianceStandards.ISO_STANDARDS['max_rotor_diameter_mm']}mm")

    if not iso_checks['pad_thickness_compliant']:
        recommendations.append(
            f"Adjust pad thickness to {ComplianceStandards.ISO_STANDARDS['min_pad_thickness_mm']}-{ComplianceStandards.ISO_STANDARDS['max_pad_thickness_mm']}mm range")

    if not indian_checks['monsoon_ready']:
        recommendations.append(
            "Consider sintered or ceramic pads for better monsoon performance")

    if not indian_checks['dust_resistant']:
        recommendations.append(
            "Upgrade to sintered or ceramic pads for Indian road conditions")

    return {
        'iso_compliance': iso_checks,
        'indian_compliance': indian_checks,
        'recommendations': recommendations,
        'overall_compliant': all(iso_checks.values()) and all(indian_checks.values())
    }

# AI-powered parameter optimization (existing)


def get_ai_optimization_suggestions(brake_params: Dict[str, Any], compliance_results: Dict[str, Any],
                                    bike_specs: Dict[str, Any]) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {'optimized_params': brake_params, 'ai_explanation': 'AI optimization unavailable - no API key'}

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""As a brake engineering expert, optimize these motorcycle brake parameters for Indian road conditions:

        Current Parameters: {brake_params}
        Bike Specifications: {bike_specs}
        Compliance Issues: {compliance_results['recommendations']}

        Provide optimized parameter values in this exact JSON format:
        {{
            "rotor_diameter_mm": ,
            "pad_thickness_mm": ,
            "pad_type": "",
            "optimization_explanation": ""
        }}

        Focus on:
        1. Meeting ISO and Indian standards
        2. Optimal performance for Indian weather conditions
        3. Durability for dusty roads
        4. Monsoon performance
        """

        response = model.generate_content(prompt)

        # Simple optimization if AI response parsing fails
        optimized_params = brake_params.copy()
        ai_explanation = "AI-optimized parameters for Indian conditions"

        # Apply basic optimizations based on compliance
        if not compliance_results['overall_compliant']:
            # Adjust rotor diameter if needed
            if brake_params['rotor_diameter_mm'] < 160:
                optimized_params['rotor_diameter_mm'] = 160

            # Upgrade pad type for Indian conditions
            if brake_params['pad_type'] == 'organic':
                optimized_params['pad_type'] = 'sintered'
                ai_explanation += " - Upgraded to sintered pads for better durability"

        return {
            'optimized_params': optimized_params,
            'ai_explanation': ai_explanation,
            'ai_response': response.text
        }

    except Exception as e:
        return {
            'optimized_params': brake_params,
            'ai_explanation': f'AI optimization failed: {str(e)}',
            'ai_response': ''
        }

# Temperature simulation with enhanced physics (existing - with updated RUL cap)


def simulate_temperature_and_wear(brake_params: Dict[str, Any], weather: str,
                                  wheel: str, driving_style: str, distance_km: int = 25000) -> Dict[str, Any]:
    # Initial conditions
    ambient_temp = BrakeConstants.AMBIENT_TEMPS[weather]
    current_temp = ambient_temp + 10
    pad_type = brake_params['pad_type']

    # Simulation parameters
    steps = BrakeConstants.TIME_STEPS
    distance_per_step = distance_km / steps

    # Physical properties
    rotor_diameter_m = brake_params['rotor_diameter_mm'] / 1000
    radiative_area = brake_params['radiative_area_m2']
    pad_mass = brake_params['pad_mass_kg']

    # Wheel distribution
    wheel_factor = 0.7 if wheel == 'front' else 0.3

    # Driving style parameters
    if driving_style == 'aggressive':
        brake_events_per_step = 3
        braking_intensity = 0.8
    else:
        brake_events_per_step = 2
        braking_intensity = 0.5

    # Weather factors
    weather_factor = BrakeConstants.WEATHER_FACTORS[weather]
    base_wear_rate = BrakeConstants.MATERIAL_LIMITS[pad_type]['wear_rate'] / 1000

    # Data storage
    distances = []
    temperatures = []
    wear_values = []
    total_wear_mm = 0

    for step in range(steps + 1):
        current_distance = step * distance_per_step
        distances.append(current_distance)

        if step > 0:
            for event in range(brake_events_per_step):
                # Velocity calculation
                velocity = brake_params['test_velocity_ms'] * \
                    min(1.0, step / steps)

                # Heat generation
                energy_dissipated = 0.5 * velocity**2 * braking_intensity * wheel_factor
                temp_rise = energy_dissipated / BrakeConstants.C_P
                current_temp += temp_rise * weather_factor['temp_mult']

                # Convective cooling
                h_conv = 8.318 * (velocity ** 0.8) if velocity > 0 else 1.0
                convective_loss = h_conv * radiative_area * \
                    (current_temp - ambient_temp) * 0.1

                # Radiative cooling
                temp_kelvin = current_temp + 273.15
                ambient_kelvin = ambient_temp + 273.15
                radiative_loss = (BrakeConstants.SIGMA * BrakeConstants.EPSILON * radiative_area *
                                  (temp_kelvin**4 - ambient_kelvin**4) * 0.1)

                # Temperature update
                total_heat_loss = convective_loss + radiative_loss
                if pad_mass > 0:
                    current_temp -= total_heat_loss / \
                        (pad_mass * BrakeConstants.C_P)
                current_temp = max(current_temp, ambient_temp)

                # Wear calculation with temperature dependency
                temp_factor = 1.0 + max(0, (current_temp - 100) / 100)
                step_wear_mm = (base_wear_rate * distance_per_step * braking_intensity *
                                wheel_factor * temp_factor * weather_factor['wear_mult'])
                total_wear_mm += step_wear_mm

        temperatures.append(current_temp)
        wear_values.append(total_wear_mm)

    # Calculate RUL with correction factor and updated upper bound
    usable_thickness = 8.0
    if total_wear_mm > 0:
        wear_rate_per_km = total_wear_mm / distance_km
        remaining_life_km = (usable_thickness -
                             total_wear_mm) / wear_rate_per_km
        # Updated to use MAX_RUL_KM = 30000
        rul_km = max(8000, min(BrakeConstants.MAX_RUL_KM, int(
            remaining_life_km / BrakeConstants.RUL_CORRECTION_FACTOR)))
    else:
        rul_km = int(40000 / BrakeConstants.RUL_CORRECTION_FACTOR)

    df = pd.DataFrame({
        'distance_km': distances,
        'temperature_c': temperatures,
        'pad_wear_mm': wear_values
    })

    return {
        'dataframe': df,
        'rul_km': rul_km,
        'max_temperature': max(temperatures),
        'final_wear_mm': total_wear_mm,
        'scenario': {
            'weather': weather,
            'wheel': wheel,
            'driving': driving_style
        }
    }

# Enhanced simulation for different speeds (existing - UNCHANGED)


def simulate_temp_vs_wear_different_speeds(brake_params: Dict[str, Any], weather: str = 'summer') -> Dict[str, Any]:
    """Simulate temperature vs wear for different speeds as shown in the provided graph"""
    max_speed_kmh = brake_params.get('test_velocity_ms', 20) * 3.6

    # Divide max speed into 3 parts as requested
    speed_1 = max_speed_kmh * 0.33  # 33% of max speed
    speed_2 = max_speed_kmh * 0.67  # 67% of max speed
    speed_3 = max_speed_kmh  # 100% of max speed

    speeds_kmh = [speed_1, speed_2, speed_3]
    speeds_ms = [s / 3.6 for s in speeds_kmh]

    ambient_temp = BrakeConstants.AMBIENT_TEMPS[weather]
    pad_type = brake_params['pad_type']
    base_wear_rate = BrakeConstants.MATERIAL_LIMITS[pad_type]['wear_rate']

    results = {}
    for i, (speed_kmh, speed_ms) in enumerate(zip(speeds_kmh, speeds_ms)):
        temperatures = []
        wear_values = []

        # Temperature range from ambient to material limit
        temp_range = np.linspace(ambient_temp,
                                 BrakeConstants.MATERIAL_LIMITS[pad_type]['max_temp'] * 0.8, 20)

        for temp in temp_range:
            # Calculate wear based on temperature and speed
            speed_factor = (speed_ms / 20) ** 1.5  # Non-linear speed effect
            # Exponential temp effect
            temp_factor = 1.0 + ((temp - ambient_temp) / 100) ** 1.2
            wear_mg = base_wear_rate * speed_factor * \
                temp_factor * 10  # Convert to mg scale

            temperatures.append(temp)
            wear_values.append(wear_mg)

        results[f"{speed_kmh:.0f}_kmh"] = {
            'temperatures': temperatures,
            'wear_values': wear_values,
            'speed_kmh': speed_kmh,
            'speed_ms': speed_ms
        }

    return results

# Comprehensive simulation runner (existing)


def run_comprehensive_simulation(brake_params: Dict[str, Any], distance_km: int = 25000) -> Dict[str, Any]:
    results = {}

    # Updated weather conditions
    weather_conditions = ['summer', 'winter', 'rainy', 'autumn']
    wheels = ['front', 'rear']
    driving_styles = ['safe', 'aggressive']

    for weather in weather_conditions:
        for wheel in wheels:
            for driving in driving_styles:
                scenario_key = f"{weather}_{wheel}_{driving}"
                result = simulate_temperature_and_wear(
                    brake_params, weather, wheel, driving, distance_km
                )
                results[scenario_key] = result

    return results

# Visualization functions (existing)


def create_weather_specific_charts(simulation_results: Dict[str, Any], selected_weather: str = None) -> Dict[str, go.Figure]:
    """Create charts for specific weather condition"""
    charts = {}
    weather_conditions = [selected_weather] if selected_weather else [
        'summer', 'winter', 'rainy', 'autumn']

    for weather in weather_conditions:
        # Temperature chart for this weather
        temp_fig = go.Figure()
        wear_fig = go.Figure()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        color_idx = 0

        for scenario_key, result in simulation_results.items():
            if not scenario_key.startswith(weather):
                continue

            df = result['dataframe']
            scenario = result['scenario']
            color = colors[color_idx % len(colors)]
            color_idx += 1

            label = f"{scenario['wheel'].title()} - {scenario['driving'].title()}"

            # Temperature trace
            temp_fig.add_trace(go.Scatter(
                x=df['distance_km'],
                y=df['temperature_c'],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=6)
            ))

            # Wear trace
            wear_fig.add_trace(go.Scatter(
                x=df['distance_km'],
                y=df['pad_wear_mm'],
                mode='lines+markers',
                name=f"{label} (RUL: {result['rul_km']:,} km)",
                line=dict(color=color, width=3),
                marker=dict(size=6)
            ))

        # Update layouts with larger height
        temp_fig.update_layout(
            title=f"Brake Temperature Analysis - {weather.title()} Season",
            xaxis_title="Distance (km)",
            yaxis_title="Temperature (°C)",
            height=600,  # Increased height
            showlegend=True
        )

        wear_fig.update_layout(
            title=f"Brake Pad Wear Analysis - {weather.title()} Season",
            xaxis_title="Distance (km)",
            yaxis_title="Pad Wear (mm)",
            height=600,  # Increased height
            showlegend=True
        )

        charts[f"{weather}_temperature"] = temp_fig
        charts[f"{weather}_wear"] = wear_fig

    return charts

# UNCHANGED: Keep the original brake pad wear vs temperature chart exactly as it is


def create_speed_based_temp_wear_chart(brake_params: Dict[str, Any]) -> go.Figure:
    """Create temperature vs wear chart for different speeds - UNCHANGED"""
    # Get simulation data for different speeds
    speed_data = simulate_temp_vs_wear_different_speeds(brake_params)

    fig = go.Figure()

    # Red, Green, Blue for the three speeds
    colors = ['#FF0000', '#00FF00', '#0000FF']
    markers = ['square', 'circle', 'diamond']

    for i, (speed_key, data) in enumerate(speed_data.items()):
        speed_kmh = data['speed_kmh']

        fig.add_trace(go.Scatter(
            x=data['temperatures'],
            y=data['wear_values'],
            mode='lines+markers',
            name=f"{speed_kmh:.0f} km/h Initial Velocity",
            line=dict(color=colors[i], width=3),
            marker=dict(symbol=markers[i], size=8),
            hovertemplate=f"Speed: {speed_kmh:.0f} km/h<br>Temperature: %{{x:.1f}}°C<br>Wear: %{{y:.2f}} mg<extra></extra>"
        ))

    fig.update_layout(
        title="Average Weight Loss per Braking vs Temperature for Different Initial Velocities",
        xaxis_title="Temperature (°C)",
        yaxis_title="Brake Pad Wear (mg)",
        height=700,  # Increased height
        showlegend=True,
        template='plotly_dark'  # Match the dark theme from the provided graph
    )

    return fig


def create_pentagon_performance_chart(brake_params: Dict[str, Any], simulation_results: Dict[str, Any]) -> go.Figure:
    """Create pentagon radar chart for overall brake performance"""
    if not simulation_results:
        return go.Figure()

    # Calculate performance metrics
    all_temps = [result['max_temperature']
        for result in simulation_results.values()]
    all_ruls = [result['rul_km'] for result in simulation_results.values()]
    all_wear = [result['final_wear_mm']
        for result in simulation_results.values()]

    # Fixed pentagon metrics calculation (0-100 scale)
    thermal_management = max(
        0, min(100, 100 - (np.mean(all_temps) - 50) * 0.8))
    durability = max(0, min(100, (np.mean(all_ruls) - 8000) / 400))

    # Fixed wear resistance calculation
    avg_wear_rate = np.mean(all_wear) / 25000  # Normalize by distance
    wear_resistance = max(
        0, min(100, 100 - (avg_wear_rate * 1000)))  # Scale properly

    stopping_power = max(
        0, min(100, brake_params.get('brake_force_n', 0) / 20))
    weather_performance = 85 if brake_params.get('pad_type') == 'sintered' else (
        90 if brake_params.get('pad_type') == 'ceramic' else 70)

    metrics = {
        'Thermal Management': thermal_management,
        'Durability': durability,
        'Wear Resistance': wear_resistance,
        'Stopping Power': stopping_power,
        'Weather Performance': weather_performance
    }

    categories = list(metrics.keys())
    values = list(metrics.values())

    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Brake Performance',
        line=dict(color='#FF6B6B', width=3),
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20%', '40%', '60%', '80%', '100%']
            )
        ),
        showlegend=False,
        title="Brake System Performance Pentagon",
        height=600  # Increased height
    )

    return fig

# PDF Report Generation (existing)


def generate_pdf_report(bike_specs: Dict[str, Any], brake_params: Dict[str, Any],
                        simulation_results: Dict[str, Any], compliance_results: Dict[str, Any]) -> bytes:
    """Generate comprehensive PDF report"""
    # Create a simple text-based report (for demonstration)
    report_content = f"""
ByteBrake AI - COMPREHENSIVE BRAKE ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MOTORCYCLE SPECIFICATIONS:
- Wheel Diameter: {bike_specs.get('wheel_diameter_inch', 'N/A')} inches
- Total Weight: {bike_specs.get('bike_weight_kg', 'N/A')} kg
- Max Speed: {bike_specs.get('max_speed_kmh', 'N/A')} km/h
- Engine CC: {bike_specs.get('engine_cc', 'N/A')} CC
- Horsepower: {bike_specs.get('horsepower', 'N/A')} HP

BRAKE PARAMETERS:
- Pad Type: {brake_params.get('pad_type', 'N/A').title()}
- Rotor Diameter: {brake_params.get('rotor_diameter_mm', 'N/A')} mm
- Pad Thickness: {brake_params.get('pad_thickness_mm', 'N/A')} mm
- Brake Force: {brake_params.get('brake_force_n', 'N/A')} N
- Stopping Distance: {brake_params.get('stopping_distance_m', 'N/A')} m

PERFORMANCE ANALYSIS:
"""

    if simulation_results:
        avg_temp = np.mean([r['max_temperature']
                           for r in simulation_results.values()])
        avg_rul = np.mean([r['rul_km'] for r in simulation_results.values()])

        report_content += f"""
- Average Maximum Temperature: {avg_temp:.1f}°C
- Average RUL: {avg_rul:,.0f} km
- Scenarios Analyzed: {len(simulation_results)}
"""

    report_content += f"""
COMPLIANCE STATUS:
- ISO Standards: {'PASS' if compliance_results.get('overall_compliant') else 'FAIL'}
- Indian Standards: {'PASS' if compliance_results.get('overall_compliant') else 'FAIL'}

RECOMMENDATIONS:
"""

    for rec in compliance_results.get('recommendations', []):
        report_content += f"- {rec}\n"

    # Convert to bytes (simplified - in production, use proper PDF library)
    return report_content.encode('utf-8')

# Conversational AI Agent (existing)


class GeniusBrakeAI:
    def __init__(self):
        self.model = None
        self.bike_specs = {}
        self.conversation_step = 0
        self.specs_complete = False

        # Initialize Gemini if available
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except:
                self.model = None

        # Conversation flow questions
        self.questions = [
            {'key': 'wheel_diameter_inch', 'question': 'What is your motorcycle wheel diameter in inches?',
             'validation': (14, 24), 'help': 'Common sizes: 17", 18", 19", 21"'},
            {'key': 'bike_weight_kg', 'question': 'What is the total weight (bike + rider) in kg?',
             'validation': (120, 400), 'help': 'Include motorcycle weight plus rider weight'},
            {'key': 'max_speed_kmh', 'question': 'What is your bike maximum speed in km/h?',
             'validation': (80, 300), 'help': 'This is used for brake system sizing'},
            {'key': 'engine_cc', 'question': 'What is your engine displacement in CC?',
             'validation': (100, 1200), 'help': 'Engine size affects brake pad material selection'},
            {'key': 'horsepower', 'question': 'What is your engine horsepower (HP)?',
             'validation': (8, 150), 'help': 'Engine power affects braking requirements'}
        ]

    def get_greeting(self) -> str:
        return """Welcome to ByteBrake AI - Advanced Brake System Engineering Assistant

I will help you analyze your motorcycle brake system with:

- Advanced temperature calculations based on heat transfer physics
- Speed vs Distance analysis with brake temperature correlation
- NEW: Dedicated Temperature vs Distance analysis
- Compliance validation for ISO and Indian standards
- AI-powered parameter optimization
- Seasonal weather analysis for Indian conditions
- Comprehensive performance reporting

Let me collect some information about your motorcycle. I need 5 specifications."""

    def process_input(self, user_input: str) -> str:
        if self.conversation_step >= len(self.questions):
            return self.handle_general_question(user_input)

        current_q = self.questions[self.conversation_step]
        try:
            import re
            numbers = re.findall(r'\d+\.?\d*', user_input)
            if not numbers:
                return f"Please provide a numeric value. {current_q['help']}\n\n{current_q['question']}"

            value = float(numbers[0])
            min_val, max_val = current_q['validation']
            if not (min_val <= value <= max_val):
                return f"Please enter a value between {min_val} and {max_val}.\n\n{current_q['question']}"

            self.bike_specs[current_q['key']] = value
            self.conversation_step += 1

            if self.conversation_step >= len(self.questions):
                self.specs_complete = True
                return self.complete_collection()
            else:
                next_q = self.questions[self.conversation_step]
                return f"Got it: {value}\n\nQuestion {self.conversation_step + 1}/5: {next_q['question']}"

        except:
            return f"Please provide a valid number. {current_q['help']}\n\n{current_q['question']}"

    def complete_collection(self) -> str:
        # Update session state
        st.session_state['bike_specs'] = self.bike_specs.copy()
        st.session_state['specs_collected'] = True
        # Auto-redirect flag
        st.session_state['auto_redirect_analytics'] = True

        summary = "Excellent! All specifications collected:\n\n"
        labels = ['Wheel Diameter', 'Total Weight',
            'Max Speed', 'Engine CC', 'Horsepower']
        units = ['inches', 'kg', 'km/h', 'CC', 'HP']

        for i, (spec, value) in enumerate(self.bike_specs.items()):
            summary += f"- {labels[i]}: {value} {units[i]}\n"

        summary += "\nCalculating brake parameters and running analysis... Redirecting to Analytics tab."
        return summary

    def handle_general_question(self, user_input: str) -> str:
        if not self.model:
            return "Your specifications are complete. Please check the Analytics section for detailed brake analysis results."

        try:
            context = f"""You are a motorcycle brake engineering expert specializing in Indian road conditions.
            User bike specs: {self.bike_specs}
            User question: {user_input}
            Provide helpful brake engineering advice considering:
            - Indian weather conditions (summer, monsoon, winter)
            - Road conditions (dust, traffic, hills)
            - Safety standards and compliance
            """

            response = self.model.generate_content(context)
            return response.text
        except:
            return "I can help with brake engineering questions. Please check the Analytics section for detailed analysis results."

    def get_current_question(self) -> str:
        if self.conversation_step < len(self.questions):
            q = self.questions[self.conversation_step]
            return f"Question {self.conversation_step + 1}/5: {q['question']}"
        return "All questions completed!"

# Main Streamlit Application


def main():
    st.title("ByteBrake AI - Advanced Brake System Engineering Assistant")
    st.markdown(
        "Comprehensive brake analysis with physics-based temperature calculations and compliance validation")

    # Initialize session state
    if 'ai_agent' not in st.session_state:
        st.session_state.ai_agent = GeniusBrakeAI()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'bike_specs' not in st.session_state:
        st.session_state.bike_specs = {}
    if 'specs_collected' not in st.session_state:
        st.session_state.specs_collected = False
    if 'brake_params' not in st.session_state:
        st.session_state.brake_params = {}
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = {}
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = {}
    if 'optimized_params' not in st.session_state:
        st.session_state.optimized_params = {}
    if 'auto_redirect_analytics' not in st.session_state:
        st.session_state.auto_redirect_analytics = False
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = 0
    if 'workshop_params' not in st.session_state:
        st.session_state.workshop_params = {}

    # Auto-redirect to Analytics tab
    if st.session_state.auto_redirect_analytics:
        st.session_state.selected_tab = 1  # Analytics tab
        st.session_state.auto_redirect_analytics = False

    # Sidebar (removed RUL correction factor display)
    with st.sidebar:
        st.header("System Status")
        if GEMINI_API_KEY and st.session_state.ai_agent.model:
            st.success("✅ AI Agent: Ready")
        else:
            st.warning("⚠️ AI Agent: Limited (No API key)")

        st.header("Progress Tracker")
        progress_items = [
            ("Specifications", st.session_state.specs_collected),
            ("Brake Parameters", len(st.session_state.brake_params) > 0),
            ("Compliance Check", len(st.session_state.compliance_results) > 0),
            ("Simulation", len(st.session_state.simulation_results) > 0)
        ]

        for item, completed in progress_items:
            if completed:
                st.success(f"✓ {item}")
            else:
                st.info(f"○ {item}")

    # Main interface with tabs (including Workshop)
    tabs = st.tabs(["💬 Conversation", "📊 Analytics",
                   "✅ Validation & Optimization", "🔧 Workshop", "📄 Reports"])

    # Auto-select Analytics tab if redirect is triggered
    selected_tab_index = st.session_state.selected_tab if st.session_state.auto_redirect_analytics else None

    with tabs[0]:
        st.header("Conversational Interface")

        # Display greeting if first time
        if not st.session_state.chat_history:
            greeting = st.session_state.ai_agent.get_greeting()
            st.session_state.chat_history.append(
                {"role": "assistant", "content": greeting})

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        # Chat input
        if user_input := st.chat_input("Type your message..."):
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input})
            with st.spinner("Processing..."):
                response = st.session_state.ai_agent.process_input(user_input)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response})

            # Check for auto-redirect
            if st.session_state.auto_redirect_analytics:
                st.rerun()
            else:
                st.rerun()

    with tabs[1]:
        st.header("Brake System Analytics")

        # Auto-calculate brake parameters if specs are collected
        if st.session_state.specs_collected and not st.session_state.brake_params:
            with st.spinner("Calculating brake parameters..."):
                st.session_state.brake_params = calculate_brake_parameters(
                    st.session_state.bike_specs)
            st.success("Brake parameters calculated!")

        # Display brake parameters
        if st.session_state.brake_params:
            st.subheader("Calculated Brake Parameters")
            col1, col2, col3, col4 = st.columns(4)

            params = st.session_state.brake_params
            with col1:
                st.metric("Rotor Diameter",
                          f"{params.get('rotor_diameter_mm', 0):.1f} mm")
                st.metric("Pad Type", params.get(
                    'pad_type', 'Unknown').title())

            with col2:
                st.metric("Pad Thickness",
                          f"{params.get('pad_thickness_mm', 0):.1f} mm")
                st.metric("Brake Force",
                          f"{params.get('brake_force_n', 0):.0f} N")

            with col3:
                st.metric("Stopping Distance",
                          f"{params.get('stopping_distance_m', 0):.1f} m")
                st.metric("Test Velocity",
                          f"{params.get('test_velocity_ms', 0)*3.6:.0f} km/h")

            with col4:
                st.metric("Radiative Area",
                          f"{params.get('radiative_area_m2', 0):.4f} m²")
                st.metric(
                    "Pad Mass", f"{params.get('pad_mass_kg', 0)*1000:.1f} g")

            # NEW: Speed vs Distance Analysis Section (existing)
            st.subheader("🚀 Speed vs Distance Analysis with Brake Temperature")
            st.markdown(
                "Generate speed patterns for user-specified distance ranges and analyze corresponding brake temperatures.")

            # Controls for speed pattern generation
            col1, col2, col3 = st.columns(3)

            with col1:
                distance_range = st.number_input("Distance Range (km)",
                                                 min_value=100, max_value=50000, value=10000, step=500,
                                                 help="Enter the distance range you want to analyze (e.g., 10000 km or 25000 km)")

            with col2:
                riding_style = st.selectbox("Riding Style",
                                            options=[
                                                'mixed', 'city_traffic', 'highway', 'aggressive'],
                                            help="Select riding pattern for speed simulation")

            with col3:
                if st.button("🔬 Generate Speed Analysis", type="primary"):
                    with st.spinner(f"Generating speed pattern for {distance_range:,} km with {riding_style} riding style..."):
                        # Create the speed vs distance chart with temperature correlation
                        speed_chart = create_speed_distance_temperature_chart(
                            distance_range, riding_style, st.session_state.brake_params
                        )

                        st.session_state['speed_chart'] = speed_chart
                        st.session_state['speed_analysis_done'] = True
                        st.success(
                            f"✅ Speed analysis complete for {distance_range:,} km!")

            # Display speed analysis chart if generated
            if st.session_state.get('speed_analysis_done', False) and 'speed_chart' in st.session_state:
                st.plotly_chart(
                    st.session_state['speed_chart'], use_container_width=True)

                # Speed pattern insights
                st.markdown("### 📊 Speed Pattern Insights")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.info(
                        f"**Riding Style:** {riding_style.replace('_', ' ').title()}")

                with col2:
                    base_speed = SPEED_KNOWLEDGE_BASE[riding_style]['base_speed']
                    st.info(f"**Base Speed:** {base_speed} km/h")

                with col3:
                    speed_var = SPEED_KNOWLEDGE_BASE[riding_style]['speed_variation']
                    st.info(f"**Speed Variation:** ±{speed_var} km/h")

            # NEW: Temperature vs Distance Analysis Section
            st.subheader("🌡️ Temperature vs Distance Analysis")
            st.markdown(
                "Dedicated temperature analysis showing how brake temperature varies with distance across different seasons.")

            # Controls for temperature analysis
            col1, col2, col3 = st.columns(3)

            with col1:
                temp_distance_range = st.number_input("Distance for Temp Analysis (km)",
                                                      min_value=100, max_value=50000, value=5000, step=500,
                                                      help="Distance range for temperature analysis",
                                                      key="temp_distance")

            with col2:
                temp_riding_style = st.selectbox("Riding Style for Temp Analysis",
                                                 options=[
                                                     'mixed', 'city_traffic', 'highway', 'aggressive'],
                                                 help="Select riding pattern for temperature simulation",
                                                 key="temp_riding_style")

            with col3:
                if st.button("🌡️ Generate Temperature Analysis", type="primary"):
                    with st.spinner(f"Generating temperature analysis for {temp_distance_range:,} km..."):
                        # Create the temperature vs distance chart
                        temp_chart = create_temperature_distance_chart(
                            temp_distance_range, temp_riding_style, st.session_state.brake_params
                        )

                        st.session_state['temp_chart'] = temp_chart
                        st.session_state['temp_analysis_done'] = True
                        st.success(
                            f"✅ Temperature analysis complete for {temp_distance_range:,} km!")

            # Display temperature analysis chart if generated
            if st.session_state.get('temp_analysis_done', False) and 'temp_chart' in st.session_state:
                st.plotly_chart(
                    st.session_state['temp_chart'], use_container_width=True)
                gif_path = "fea.gif"
                if os.path.exists(gif_path):
                    st.markdown("### 🎬 FEA Animation Visualization")
                    st.markdown("Finite Element Analysis animation showing brake temperature distribution")
                    
                    # Display the GIF
                    with open(gif_path, "rb") as gif_file:
                        gif_bytes = gif_file.read()
                    st.image(gif_bytes, caption="FEA Temperature Distribution Animation", use_column_width=True)
                else:
                    st.warning("⚠️ fea.gif not found in the application directory. Please ensure the file is placed in the same folder as your Streamlit app.")

                # Temperature analysis insights
                st.markdown("### 🌡️ Temperature Analysis Insights")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    summer_temp = BrakeConstants.AMBIENT_TEMPS['summer']
                    st.info(f"**Summer Base:** {summer_temp}°C")

                with col2:
                    winter_temp = BrakeConstants.AMBIENT_TEMPS['winter']
                    st.info(f"**Winter Base:** {winter_temp}°C")

                with col3:
                    rainy_temp = BrakeConstants.AMBIENT_TEMPS['rainy']
                    st.info(f"**Rainy Base:** {rainy_temp}°C")

                with col4:
                    autumn_temp = BrakeConstants.AMBIENT_TEMPS['autumn']
                    st.info(f"**Autumn Base:** {autumn_temp}°C")

            # Seasonal tabs for weather analysis (existing)
            st.subheader("Seasonal Performance Analysis")

            # Sub-tabs for different weather conditions
            weather_tabs = st.tabs(
                ["☀️ Summer", "❄️ Winter", "🌧️ Rainy", "🍂 Autumn", "📈 Combined View"])

            # Simulation controls
            col1, col2 = st.columns(2)

            with col1:
                test_distance = st.number_input("Test Distance (km)",
                                                min_value=1000, max_value=50000, value=25000, step=1000)

            with col2:
                if st.button("Run Comprehensive Analysis", type="primary"):
                    with st.spinner("Running advanced brake simulations for all seasons..."):
                        st.session_state.simulation_results = run_comprehensive_simulation(
                            st.session_state.brake_params, test_distance
                        )
                    st.success(
                        f"Analysis complete! {len(st.session_state.simulation_results)} scenarios analyzed.")

            # Weather-specific analysis in sub-tabs
            if st.session_state.simulation_results:
                seasons = ['summer', 'winter', 'rainy', 'autumn']

                for i, season in enumerate(seasons):
                    with weather_tabs[i]:
                        st.markdown(f"### {season.title()} Season Analysis")

                        # Create charts for this specific weather
                        weather_charts = create_weather_specific_charts(
                            st.session_state.simulation_results, season)

                        # Display temperature chart (full width, vertical)
                        temp_key = f"{season}_temperature"
                        if temp_key in weather_charts:
                            st.plotly_chart(
                                weather_charts[temp_key], use_container_width=True)

                        # Display wear chart (full width, vertical)
                        wear_key = f"{season}_wear"
                        if wear_key in weather_charts:
                            st.plotly_chart(
                                weather_charts[wear_key], use_container_width=True)

                # Combined view tab
                with weather_tabs[4]:
                    st.markdown(
                        "### Temperature vs Brake Pad Wear (Different Initial Velocities)")
                    speed_wear_chart = create_speed_based_temp_wear_chart(
                        st.session_state.brake_params)
                    st.plotly_chart(speed_wear_chart, use_container_width=True)

                # Results summary table
                st.subheader("Performance Summary by Season")
                summary_data = []
                for scenario_key, result in st.session_state.simulation_results.items():
                    scenario = result['scenario']
                    summary_data.append({
                        'Season': scenario['weather'].title(),
                        'Wheel': scenario['wheel'].title(),
                        'Driving': scenario['driving'].title(),
                        'Max Temperature (°C)': f"{result['max_temperature']:.1f}",
                        'Final Wear (mm)': f"{result['final_wear_mm']:.3f}",
                        'RUL (km)': f"{result['rul_km']:,}"
                    })

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

        # Manual specification input fallback
        elif not st.session_state.specs_collected:
            st.info(
                "Please complete the conversation in the first tab to collect motorcycle specifications.")

    # Keep all remaining tabs exactly as they were (Validation & Optimization, Workshop, Reports)
    with tabs[2]:
        st.header("Compliance Validation & AI Optimization")

        if st.session_state.brake_params:
            # Run compliance validation
            if not st.session_state.compliance_results:
                with st.spinner("Validating compliance with ISO and Indian standards..."):
                    st.session_state.compliance_results = validate_brake_compliance(
                        st.session_state.brake_params)

            # Display compliance results
            st.subheader("Standards Compliance Check")
            compliance = st.session_state.compliance_results

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**ISO Standards**")
                for check, passed in compliance['iso_compliance'].items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    st.write(f"{status} {clean_for_display(check)}")

            with col2:
                st.markdown("**Indian Standards**")
                for check, passed in compliance['indian_compliance'].items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    st.write(f"{status} {clean_for_display(check)}")

            # Overall compliance status
            if compliance['overall_compliant']:
                st.success("🎉 All compliance checks passed!")
            else:
                st.error(
                    "⚠️ Some compliance issues found. See recommendations below.")

            # Display recommendations
            if compliance['recommendations']:
                st.subheader("Compliance Recommendations")
                for rec in compliance['recommendations']:
                    st.write(f"• {rec}")

            # AI Optimization
            st.subheader("AI-Powered Parameter Optimization")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Get AI Optimization Suggestions", type="primary"):
                    with st.spinner("AI is analyzing and optimizing brake parameters..."):
                        optimization = get_ai_optimization_suggestions(
                            st.session_state.brake_params,
                            compliance,
                            st.session_state.bike_specs
                        )
                        st.session_state.optimized_params = optimization

            with col2:
                if st.session_state.optimized_params and st.button("Apply Optimized Parameters"):
                    st.session_state.brake_params.update(
                        st.session_state.optimized_params['optimized_params'])
                    st.session_state.compliance_results = {}  # Reset to re-validate
                    st.session_state.simulation_results = {}  # Reset to re-run
                    st.success("Parameters updated! Please re-run analytics.")
                    st.rerun()

            # Display AI optimization results
            if st.session_state.optimized_params:
                st.subheader("AI Optimization Results")
                opt = st.session_state.optimized_params

                st.info(
                    f"**AI Explanation:** {opt.get('ai_explanation', 'No explanation available')}")

                if 'ai_response' in opt and opt['ai_response']:
                    with st.expander("Full AI Response"):
                        st.write(opt['ai_response'])

                # Show parameter comparison
                if 'optimized_params' in opt:
                    st.subheader("Parameter Comparison")
                    comparison_data = []
                    for key in st.session_state.brake_params.keys():
                        original = st.session_state.brake_params.get(
                            key, 'N/A')
                        optimized = opt['optimized_params'].get(key, 'N/A')
                        comparison_data.append({
                            'Parameter': clean_for_display(key),
                            'Original': str(original),
                            'Optimized': str(optimized),
                            'Changed': '✓' if original != optimized else ''
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

        else:
            st.info(
                "Please complete brake parameter calculation in the Analytics tab first.")

    with tabs[3]:
        st.header("Workshop - Interactive Parameter Tuning")

        if st.session_state.brake_params:
            st.markdown(
                "### Adjust brake parameters using sliders to see real-time impact on performance")

            # Parameter sliders
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Brake Pad Parameters")

                # Pad type selector
                pad_types = ['organic', 'sintered', 'ceramic']
                current_pad_type = st.session_state.brake_params.get(
                    'pad_type', 'sintered')
                new_pad_type = st.selectbox("Pad Type", pad_types,
                                            index=pad_types.index(current_pad_type) if current_pad_type in pad_types else 1)

                # Pad thickness slider
                current_thickness = st.session_state.brake_params.get(
                    'pad_thickness_mm', 10.0)
                new_thickness = st.slider("Pad Thickness (mm)",
                                          min_value=4.0, max_value=20.0,
                                          value=float(current_thickness), step=0.5)

                # Friction coefficient slider
                material_friction = BrakeConstants.MATERIAL_LIMITS.get(
                    new_pad_type, {}).get('friction_coeff', 0.4)
                new_friction = st.slider("Friction Coefficient",
                                         min_value=0.3, max_value=0.7,
                                         value=float(material_friction), step=0.05)

            with col2:
                st.subheader("Rotor Parameters")

                # Rotor diameter slider
                current_diameter = st.session_state.brake_params.get(
                    'rotor_diameter_mm', 200.0)
                new_diameter = st.slider("Rotor Diameter (mm)",
                                         min_value=160.0, max_value=350.0,
                                         value=float(current_diameter), step=10.0)

                # Test distance for workshop simulation
                workshop_distance = st.slider("Test Distance (km)",
                                              min_value=5000, max_value=30000,
                                              value=15000, step=1000)

            # Apply workshop parameters
            workshop_params = {
                'pad_type': new_pad_type,
                'pad_thickness_mm': new_thickness,
                'rotor_diameter_mm': new_diameter,
                'friction_coefficient': new_friction
            }

            # Calculate new brake parameters with workshop settings
            if st.button("Apply Workshop Parameters", type="primary"):
                with st.spinner("Calculating with new parameters..."):
                    st.session_state.workshop_params = calculate_brake_parameters(
                        st.session_state.bike_specs, workshop_params
                    )

                    # Run quick simulation
                    workshop_results = run_comprehensive_simulation(
                        st.session_state.workshop_params, workshop_distance
                    )

                    st.session_state.workshop_simulation = workshop_results
                    st.success("Workshop analysis complete!")

            # Display workshop results
            if st.session_state.workshop_params:
                st.markdown("### Workshop Results")

                # Compare original vs workshop parameters
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Original Parameters")
                    orig_params = st.session_state.brake_params
                    st.metric(
                        "Rotor Diameter", f"{orig_params.get('rotor_diameter_mm', 0):.1f} mm")
                    st.metric("Pad Type", orig_params.get(
                        'pad_type', 'Unknown').title())
                    st.metric(
                        "Stopping Distance", f"{orig_params.get('stopping_distance_m', 0):.1f} m")
                    st.metric("Brake Force",
                              f"{orig_params.get('brake_force_n', 0):.0f} N")

                with col2:
                    st.markdown("#### Workshop Parameters")
                    workshop_params = st.session_state.workshop_params
                    st.metric(
                        "Rotor Diameter", f"{workshop_params.get('rotor_diameter_mm', 0):.1f} mm")
                    st.metric("Pad Type", workshop_params.get(
                        'pad_type', 'Unknown').title())
                    st.metric(
                        "Stopping Distance", f"{workshop_params.get('stopping_distance_m', 0):.1f} m")
                    st.metric(
                        "Brake Force", f"{workshop_params.get('brake_force_n', 0):.0f} N")

                # Workshop simulation charts
                if 'workshop_simulation' in st.session_state:
                    st.markdown("### Workshop Performance Charts")

                    workshop_charts = create_weather_specific_charts(
                        st.session_state.workshop_simulation, 'summer'
                    )

                    # Display summer charts as example
                    if 'summer_temperature' in workshop_charts:
                        st.plotly_chart(
                            workshop_charts['summer_temperature'], use_container_width=True)

                    if 'summer_wear' in workshop_charts:
                        st.plotly_chart(
                            workshop_charts['summer_wear'], use_container_width=True)

                    # Performance comparison
                    if st.session_state.simulation_results:
                        st.markdown("### Performance Comparison")

                        # Get summer results for comparison
                        orig_summer = [r for k, r in st.session_state.simulation_results.items(
                        ) if k.startswith('summer')]
                        workshop_summer = [r for k, r in st.session_state.workshop_simulation.items(
                        ) if k.startswith('summer')]

                        if orig_summer and workshop_summer:
                            orig_avg_temp = np.mean(
                                [r['max_temperature'] for r in orig_summer])
                            workshop_avg_temp = np.mean(
                                [r['max_temperature'] for r in workshop_summer])

                            orig_avg_rul = np.mean(
                                [r['rul_km'] for r in orig_summer])
                            workshop_avg_rul = np.mean(
                                [r['rul_km'] for r in workshop_summer])

                            col1, col2 = st.columns(2)

                            with col1:
                                temp_change = workshop_avg_temp - orig_avg_temp
                                st.metric("Avg Temperature Change",
                                          f"{temp_change:+.1f}°C")

                            with col2:
                                rul_change = workshop_avg_rul - orig_avg_rul
                                st.metric("Avg RUL Change",
                                          f"{rul_change:+,.0f} km")

        else:
            st.info(
                "Please complete brake parameter calculation in the Analytics tab first.")

    with tabs[4]:
        st.header("Performance Reports & Analysis")
        if st.session_state.brake_params and st.session_state.simulation_results:
            # Pentagon Performance Chart (fixed wear resistance display)
            
                        # ===== GLB MODEL VIEWER ADDITION =====
            st.markdown("---")
            st.subheader("3D Brake Model Visualization")
            st.markdown(
                "Interactive 3D visualization of brake components with metallic materials and mesh overlay")


            def get_model_data_reports(file_path):
                """Convert GLB file to base64 for embedding"""
                try:
                    with open(file_path, "rb") as file:
                        return base64.b64encode(file.read()).decode()
                except FileNotFoundError:
                    st.error(
                        f"Model file '{file_path}' not found. Please ensure the file exists in the same directory as this app.")
                    return None


            def create_metallic_threejs_viewer_reports(model_data):
                """Create Three.js viewer with metallic materials and mesh visualization"""
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Brake Component Viewer</title>
                    <style>
                        body {{ margin: 0; padding: 0; overflow: hidden; background: #1a1a1a; }}
                        #viewer {{ width: 100vw; height: 100vh; }}
                        #controls {{
                            position: absolute;
                            top: 10px;
                            left: 10px;
                            z-index: 100;
                            background: rgba(0,0,0,0.7);
                            padding: 10px;
                            border-radius: 5px;
                            color: white;
                        }}
                        button {{
                            margin: 5px;
                            padding: 5px 10px;
                            background: #333;
                            color: white;
                            border: none;
                            border-radius: 3px;
                            cursor: pointer;
                        }}
                        button:hover {{ background: #555; }}
                    </style>
                </head>
                <body>
                    <div id="controls">
                        <button onclick="toggleWireframe()">Toggle Mesh</button>
                        <button onclick="toggleMetallic()">Toggle Metallic</button>
                        <button onclick="resetView()">Reset View</button>
                    </div>
                    <div id="viewer"></div>

                    <script type="importmap">
                    {{
                        "imports": {{
                            "three": "https://cdn.jsdelivr.net/npm/three@0.157.0/build/three.module.js",
                            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.157.0/examples/jsm/"
                        }}
                    }}
                    </script>

                    <script type="module">
                        import * as THREE from 'three';
                        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
                        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

                        let scene, camera, renderer, controls;
                        let originalModel, meshModel;
                        let isWireframeVisible = false;
                        let isMetallicMode = true;

                        scene = new THREE.Scene();
                        scene.background = new THREE.Color(0x2a2a2a);

                        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                        camera.position.set(5, 5, 5);

                        renderer = new THREE.WebGLRenderer({{
                            antialias: true,
                            alpha: true
                        }});
                        renderer.setSize(window.innerWidth, window.innerHeight);
                        renderer.shadowMap.enabled = true;
                        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                        renderer.toneMapping = THREE.ACESFilmicToneMapping;
                        renderer.toneMappingExposure = 1.2;
                        renderer.outputEncoding = THREE.sRGBEncoding;
                        document.getElementById('viewer').appendChild(renderer.domElement);

                        controls = new OrbitControls(camera, renderer.domElement);
                        controls.enableDamping = true;
                        controls.dampingFactor = 0.05;
                        controls.enableZoom = true;
                        controls.autoRotate = true;
                        controls.autoRotateSpeed = 1.0;

                        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
                        scene.add(ambientLight);

                        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.5);
                        directionalLight1.position.set(5, 5, 5);
                        directionalLight1.castShadow = true;
                        directionalLight1.shadow.mapSize.width = 2048;
                        directionalLight1.shadow.mapSize.height = 2048;
                        scene.add(directionalLight1);

                        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.8);
                        directionalLight2.position.set(-5, 3, -5);
                        scene.add(directionalLight2);

                        const directionalLight3 = new THREE.DirectionalLight(0x88ccff, 0.6);
                        directionalLight3.position.set(0, -5, 0);
                        scene.add(directionalLight3);

                        const pointLight1 = new THREE.PointLight(0xffffff, 1, 100);
                        pointLight1.position.set(10, 10, 10);
                        scene.add(pointLight1);

                        const pointLight2 = new THREE.PointLight(0xff6600, 0.8, 100);
                        pointLight2.position.set(-10, -10, 10);
                        scene.add(pointLight2);

                        const loader = new THREE.CubeTextureLoader();
                        const envMap = loader.load([
                            'https://threejs.org/examples/textures/cube/SwedishRoyalCastle/px.jpg',
                            'https://threejs.org/examples/textures/cube/SwedishRoyalCastle/nx.jpg',
                            'https://threejs.org/examples/textures/cube/SwedishRoyalCastle/py.jpg',
                            'https://threejs.org/examples/textures/cube/SwedishRoyalCastle/ny.jpg',
                            'https://threejs.org/examples/textures/cube/SwedishRoyalCastle/pz.jpg',
                            'https://threejs.org/examples/textures/cube/SwedishRoyalCastle/nz.jpg'
                        ]);
                        scene.environment = envMap;

                        const metallicMaterial = new THREE.MeshStandardMaterial({{
                            color: 0xc0c0c0,
                            metalness: 0.9,
                            roughness: 0.1,
                            envMap: envMap,
                            envMapIntensity: 1.5
                        }});

                        const wireframeMaterial = new THREE.MeshBasicMaterial({{
                            color: 0x00ffff,
                            wireframe: true,
                            transparent: true,
                            opacity: 0.8
                        }});

                        const gltfLoader = new GLTFLoader();
                        const modelData = 'data:application/octet-stream;base64,{model_data}';

                        gltfLoader.load(modelData, function(gltf) {{
                            originalModel = gltf.scene.clone();
                            meshModel = gltf.scene.clone();

                            originalModel.traverse(function(child) {{
                                if (child.isMesh) {{
                                    child.material = metallicMaterial.clone();
                                    child.castShadow = true;
                                    child.receiveShadow = true;
                                }}
                            }});

                            meshModel.traverse(function(child) {{
                                if (child.isMesh) {{
                                    child.material = wireframeMaterial.clone();
                                    child.renderOrder = 1;
                                }}
                            }});

                            const box = new THREE.Box3().setFromObject(originalModel);
                            const center = box.getCenter(new THREE.Vector3());
                            const size = box.getSize(new THREE.Vector3());
                            const maxDim = Math.max(size.x, size.y, size.z);
                            const scale = 8 / maxDim;

                            [originalModel, meshModel].forEach(model => {{
                                model.scale.multiplyScalar(scale);
                                model.position.sub(center.clone().multiplyScalar(scale));
                            }});

                            scene.add(originalModel);
                            meshModel.visible = false;
                            scene.add(meshModel);

                        }}, undefined, function(error) {{
                            console.error('Error loading GLB model:', error);
                        }});

                        window.toggleWireframe = function() {{
                            isWireframeVisible = !isWireframeVisible;
                            if (meshModel) {{
                                meshModel.visible = isWireframeVisible;
                            }}
                        }};

                        window.toggleMetallic = function() {{
                            isMetallicMode = !isMetallicMode;
                            if (originalModel) {{
                                originalModel.traverse(function(child) {{
                                    if (child.isMesh) {{
                                        if (isMetallicMode) {{
                                            child.material = metallicMaterial.clone();
                                        }} else {{
                                            child.material = new THREE.MeshLambertMaterial({{
                                                color: 0x888888
                                            }});
                                        }}
                                    }}
                                }});
                            }}
                        }};

                        window.resetView = function() {{
                            camera.position.set(5, 5, 5);
                            controls.reset();
                        }};

                        function animate() {{
                            requestAnimationFrame(animate);
                            controls.update();
                            renderer.render(scene, camera);
                        }}
                        animate();

                        window.addEventListener('resize', function() {{
                            camera.aspect = window.innerWidth / window.innerHeight;
                            camera.updateProjectionMatrix();
                            renderer.setSize(window.innerWidth, window.innerHeight);
                        }});
                    </script>
                </body>
                </html>
                """
                return html_content


            # Load and display the default GLB model
            default_model = "start_model_step.glb"

            if os.path.exists(default_model):
                with st.spinner("Loading 3D brake model..."):
                    model_data = get_model_data_reports(default_model)

                    if model_data:
                        html_content = create_metallic_threejs_viewer_reports(model_data)

                        # Display controls info
                        st.info("""
                        **3D Model Controls:**
                        - **Toggle Mesh**: Show/hide wireframe overlay
                        - **Toggle Metallic**: Switch between metallic and standard material
                        - **Reset View**: Return to default camera position
                        - **Mouse**: Left click + drag to rotate, scroll to zoom, right click + drag to pan
                        """)

                        # Display the 3D viewer
                        st.components.v1.html(html_content, height=700)
            else:
                st.error(f"Default model file '{default_model}' not found!")
                st.markdown(
                    "Please ensure that `start_model_step.glb` is in the same directory as this Streamlit app.")

            # ===== END GLB MODEL VIEWER ADDITION =====


            st.subheader("Overall Performance Pentagon")

            pentagon_chart = create_pentagon_performance_chart(
                st.session_state.brake_params,
                st.session_state.simulation_results
            )
            st.plotly_chart(pentagon_chart, use_container_width=True)

            # Performance Insights
            st.subheader("Performance Insights")

            # Calculate insights
            all_temps = [r['max_temperature']
                for r in st.session_state.simulation_results.values()]
            all_ruls = [r['rul_km']
                for r in st.session_state.simulation_results.values()]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Average Max Temperature",
                          f"{np.mean(all_temps):.1f}°C")

            with col2:
                st.metric("Average RUL", f"{np.mean(all_ruls):,.0f} km")

            with col3:
                st.metric("Best RUL Scenario", f"{max(all_ruls):,} km")

            with col4:
                st.metric("Temperature Range",
                          f"{min(all_temps):.1f}°C - {max(all_temps):.1f}°C")

            # Find best and worst performing scenarios
            st.subheader("Scenario Performance Ranking")

            scenario_performance = []
            for scenario_key, result in st.session_state.simulation_results.items():
                scenario = result['scenario']
                performance_score = (result['rul_km'] / max(all_ruls)) * 50 + (
                    1 - result['max_temperature'] / max(all_temps)) * 50

                scenario_performance.append({
                    'Scenario': f"{scenario['weather'].title()} - {scenario['wheel'].title()} - {scenario['driving'].title()}",
                    'RUL (km)': f"{result['rul_km']:,}",
                    'Max Temperature (°C)': f"{result['max_temperature']:.1f}",
                    'Performance Score': f"{performance_score:.1f}/100"
                })

            # Sort by performance score
            scenario_df = pd.DataFrame(scenario_performance)
            scenario_df['Score_Numeric'] = scenario_df['Performance Score'].str.replace(
                '/100', '').astype(float)
            scenario_df = scenario_df.sort_values(
                'Score_Numeric', ascending=False)
            scenario_df = scenario_df.drop('Score_Numeric', axis=1)

            st.dataframe(scenario_df, use_container_width=True)

            # AI Performance Analysis
            if GEMINI_API_KEY and st.session_state.ai_agent.model:
                st.subheader("AI Performance Analysis")

                if st.button("Generate AI Performance Report"):
                    with st.spinner("AI is analyzing brake system performance..."):
                        try:
                            analysis_prompt = f"""As a brake engineering expert, provide a comprehensive performance analysis:

                            Brake Parameters: {st.session_state.brake_params}
                            Bike Specifications: {st.session_state.bike_specs}
                            Performance Data:
                            - Temperature range: {min(all_temps):.1f}°C to {max(all_temps):.1f}°C
                            - RUL range: {min(all_ruls):,} to {max(all_ruls):,} km
                            - Scenarios tested: {len(st.session_state.simulation_results)}

                            Provide insights on:
                            1. Overall brake system performance
                            2. Seasonal performance variations
                            3. Recommendations for Indian road conditions
                            4. Safety considerations
                            5. Maintenance recommendations
                            """

                            response = st.session_state.ai_agent.model.generate_content(
                                analysis_prompt)
                            st.success("AI Performance Analysis:")
                            st.write(response.text)

                        except Exception as e:
                            st.error(f"AI analysis failed: {str(e)}")

            # PDF Report Generation
            st.subheader("Download Comprehensive Report")

            col1, col2 = st.columns(2)

            with col1:
                report_format = st.selectbox(
                    "Report Format", ["PDF (Text)", "Detailed Analysis"])
                include_charts = st.checkbox(
                    "Include Performance Charts", value=True)

            with col2:
                include_compliance = st.checkbox(
                    "Include Compliance Results", value=True)
                include_ai_analysis = st.checkbox(
                    "Include AI Analysis", value=True)

            if st.button("Generate & Download Report", type="primary"):
                with st.spinner("Generating comprehensive brake analysis report..."):
                    try:
                        pdf_bytes = generate_pdf_report(
                            st.session_state.bike_specs,
                            st.session_state.brake_params,
                            st.session_state.simulation_results,
                            st.session_state.compliance_results
                        )

                        # Create download button
                        st.download_button(
                            label="📥 Download Report",
                            data=pdf_bytes,
                            file_name=f"brake_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        st.success("✅ Report generated successfully!")

                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")

        else:
            st.info(
                "Please complete brake parameter calculation and simulation to generate reports.")

    # Footer
    st.markdown("---")
    st.markdown(
        "**ByteBrake AI** - Advanced Brake System Engineering with Compliance Validation")
    st.markdown(
        "Enhanced with seasonal analysis, AI optimization, interactive workshop, and speed pattern analysis")

if __name__ == "__main__":
    main()
