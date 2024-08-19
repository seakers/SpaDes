## Capabilites

def compute_image_distortion_in_side_looking_instruments(instruments):
    for instr in instruments:
        instr.image_distortion = instr.orbit_altitude / instr.characteristic_orbit

def compute_soil_penetration(instruments):
    for instr in instruments:
        instr.soil_penetration = get_soil_penetration(instr.frequency)

def get_soil_penetration(f):
    lam = 3e10 / f  # lambda in cm
    if lam < 1:
        return 0.001
    elif 1 <= lam < 2:
        return 0.01
    elif 2 <= lam < 5:
        return 0.05
    elif 5 <= lam < 10:
        return 0.08
    elif 10 <= lam < 25:
        return 0.3
    elif 25 <= lam < 50:
        return 0.8
    else:
        return 1.0

## Capabilites-Remove-Overlaps

def remove_overlapping_cross_registered_measurements(instruments):
    measurements = []
    for ins in instruments:
        instrument = ins.strip()
        measurements += get_measurements_from_instrument(instrument)

    unique_measurements = list(set(measurements))
    return unique_measurements

## CAPABILITIES-GENERATE

def generate_capabilities_measurements(instruments):
    for instrument in instruments:
        this = instrument.manifested_instrument
        this2 = instrument.can_measure

        if isinstance(this2.data_rate_duty_cycle, (int, float)) and isinstance(this2.power_duty_cycle, (int, float)):
            science_multiplier = min(this2.data_rate_duty_cycle, this2.power_duty_cycle)
        else:
            science_multiplier = 1.0

        resource_limitations = {
            'data_rate_duty_cycle': this2.data_rate_duty_cycle,
            'power_duty_cycle': this2.power_duty_cycle,
            'factHistory': "{R" + str(rulesMap.get('CAPABILITIES-GENERATE::{{instrument.name}}-measurements')) + " A" + str(this.getFactId()) + " A" + str(this2.getFactId()) + "}"
        }

        measurements = []
        for measurement in instrument.measurements:
            attributes = ""
            for attribute in measurement.attributes:
                if attribute.value.lower() != "nil":
                    attributes += "(" + attribute.key + " " + attribute.value + ") "

            measurement_data = {
                'Parameter': measurement.name,
                'science_multiplier': science_multiplier,
                'taken_by': instrument.name,
                'flies_in': this.flies_in,
                'orbit_altitude': this.orbit_altitude,
                'orbit_RAAN': this.orbit_RAAN,
                'orbit_anomaly': this.orbit_anomaly,
                'Id': instrument.name + str(measurement.index),
                'Instrument': instrument.name,
                'factHistory': "{R" + str(rulesMap.get('CAPABILITIES-GENERATE::{{instrument.name}}-measurements')) + " A" + str(this.getFactId()) + " A" + str(this2.getFactId()) + "}",
                'avg_revisit_time_cold_regions': this.avg_revisit_time_cold_regions,
                'avg_revisit_time_global': this.avg_revisit_time_global,
                'avg_revisit_time_northern_hemisphere': this.avg_revisit_time_northern_hemisphere,
                'avg_revisit_time_southern_hemisphere': this.avg_revisit_time_southern_hemisphere,
                'avg_revisit_time_tropics': this.avg_revisit_time_tropics,
                'avg_revisit_time_US': this.avg_revisit_time_US
            }

            measurements.append(measurement_data)

        cross_registered = {
            'measurements': instrument.list_of_measurements,
            'degree_of_cross_registration': 'instrument',
            'platform': this.Id,
            'factHistory': "{R" + str(rulesMap.get('CAPABILITIES-GENERATE::{{instrument.name}}-measurements')) + " A" + str(this.getFactId()) + " A" + str(this2.getFactId()) + "}"
        }

        this.modify(measurement_ids=instrument.list_of_measurements, factHistory="{R" + str(rulesMap.get('CAPABILITIES-GENERATE::{{instrument.name}}-measurements')) + " " + this.factHistory + " S" + str(this2.getFactId()) + "}")
        this2.modify(copied_to_measurement_fact='yes', factHistory="{R" + str(rulesMap.get('CAPABILITIES-GENERATE::{{instrument.name}}-measurements')) + " " + this.factHistory + " S" + str(this2.getFactId()) + "}")

        return {
            'resource_limitations': resource_limitations,
            'measurements': measurements,
            'cross_registered': cross_registered
        }

## CAPABILITIES-CROSS-REGISTER

def cross_register_measurements_from_cross_registered_instruments(instruments):
    measurements = []
    for ins in instruments:
        instrument = ins.strip()
        measurements += get_measurements_from_instrument(instrument)

    return {
        'measurements': measurements,
        'degree_of_cross_registration': 'spacecraft'
    }

def get_measurements_from_instrument(instrument):
    # Add your logic here to get the measurements from the instrument
    # and return the list of measurements
    return measurements

## CAPABILITIES-UPDATE

def update_diurnal_cycle(meas):
    if meas.diurnal_cycle is None and meas.orbit_inclination is not None and meas.orbit_RAAN is not None:
        if meas.orbit_inclination == 'polar':
            dc = 'variable'
        else:
            dc = meas.orbit_RAAN + '-only'

        meas.diurnal_cycle = dc
        meas.factHistory = "{R" + str(rulesMap.get('CAPABILITIES-UPDATE::basic-diurnal-cycle')) + " " + meas.factHistory + "}"

## SYNERGIES

# using the SMAP rules for this part because these rules are different for different missions

def SMAP_spatial_disaggregation(m1, m2, measurements):
    if m1.Parameter == "2.3.2 soil moisture" and m1.Illumination == "Active" and m1.Horizontal_Spatial_Resolution is not None and m1.Accuracy is not None and m2.Parameter == "2.3.2 soil moisture" and m2.Illumination == "Passive" and m2.Horizontal_Spatial_Resolution is not None and m2.Accuracy is not None and any(meas.Id == m1.Id or meas.Id == m2.Id for meas in measurements) and "disaggregated" not in m1.taken_by and "disaggregated" not in m2.taken_by:
        new_measurement = {
            'Parameter': m1.Parameter,
            'Illumination': m1.Illumination,
            'Horizontal_Spatial_Resolution': (m1.Horizontal_Spatial_Resolution * m2.Horizontal_Spatial_Resolution) ** 0.5,
            'Accuracy': m2.Accuracy,
            'Id': f"{m1.Id}-disaggregated-{m2.Id}",
            'taken_by': f"{m1.taken_by}-{m2.taken_by}-disaggregated"
        }
        measurements.append(new_measurement)

def carbon_net_ecosystem_exchange(SM, measurements):
    if SM.Parameter == "2.3.2 soil moisture" and any(meas.Parameter == "2.5.1 Surface temperature -land-" for meas in measurements) and any(meas.Parameter == "2.6.2 landcover status" for meas in measurements) and any(meas.Parameter == "2.4.2 vegetation state" for meas in measurements) and all(meas.Id != f"{SM.Id}-syn{meas.Id}" for meas in measurements):
        new_measurement = {
            'Parameter': "2.3.3 Carbon net ecosystem exchange NEE",
            'Id': f"{SM.Id}-syn-{meas.Id}",
            'taken_by': f"{SM.taken_by}-syn-{meas.taken_by}"
        }
        measurements.append(new_measurement)

def snow_cover_3freqs(IR, X, L, measurements):
    if IR.Parameter == "4.2.4 snow cover" and IR.Spectral_region == "opt-VNIR+TIR" and IR.Accuracy == "Low" and X.Parameter == "4.2.4 snow cover" and X.Spectral_region == "MW-X+Ka+Ku+mm" and X.Accuracy == "Low" and L.Parameter == "4.2.4 snow cover" and L.Spectral_region == "MW-L" and L.Accuracy == "Low" and all(meas.Id != f"{IR.Id}-syn-{X.Id}-syn-{L.Id}" for meas in measurements):
        new_measurement = {
            'Parameter': "4.2.4 snow cover",
            'Spectral_region': "MW-X+Ka+Ku+mm",
            'Accuracy': "High",
            'Id': f"{IR.Id}-syn-{X.Id}-syn-{L.Id}",
            'taken_by': f"{IR.taken_by}-syn-{X.taken_by}-syn-{L.taken_by}"
        }
        measurements.append(new_measurement)

def snow_cover_2freqs(IR, MW, measurements):
    if IR.Parameter == "4.2.4 snow cover" and IR.Spectral_region == "opt-VNIR+TIR" and IR.Accuracy == "Low" and MW.Parameter == "4.2.4 snow cover" and MW.Spectral_region is not None and MW.Accuracy == "Low" and "MW" in MW.Spectral_region and all(meas.Id != f"{IR.Id}-syn-{MW.Id}" for meas in measurements):
        new_measurement = {
            'Parameter': "4.2.4 snow cover",
            'Spectral_region': MW.Spectral_region,
            'Accuracy': "Medium",
            'Id': f"{IR.Id}-syn-{MW.Id}",
            'taken_by': f"{IR.taken_by}-syn-{MW.taken_by}"
        }
        measurements.append(new_measurement)

def ice_cover_3freqs(IR, X, L, measurements):
    if IR.Parameter == "4.3.2 Sea ice cover" and IR.Spectral_region == "opt-VNIR+TIR" and IR.Accuracy == "Low" and X.Parameter == "4.3.2 Sea ice cover" and X.Spectral_region == "MW-X+Ka+Ku+mm" and X.Accuracy == "Low" and L.Parameter == "4.3.2 Sea ice cover" and L.Spectral_region == "MW-L" and L.Accuracy == "Low" and all(meas.Id != f"{IR.Id}-syn-{X.Id}-syn-{L.Id}" for meas in measurements):
        new_measurement = {
            'Parameter': "4.3.2 Sea ice cover",
            'Spectral_region': "MW-X+Ka+Ku+mm",
            'Accuracy': "High",
            'Id': f"{IR.Id}-syn-{X.Id}-syn-{L.Id}",
            'taken_by': f"{IR.taken_by}-syn-{X.taken_by}-syn-{L.taken_by}"
        }
        measurements.append(new_measurement)

def ice_cover_2freqs(IR, MW, measurements):
    if IR.Parameter == "4.3.2 Sea ice cover" and IR.Spectral_region == "opt-VNIR+TIR" and IR.Accuracy == "Low" and MW.Parameter == "4.3.2 Sea ice cover" and MW.Spectral_region is not None and MW.Accuracy == "Low" and "MW" in MW.Spectral_region and all(meas.Id != f"{IR.Id}-syn-{MW.Id}" for meas in measurements):
        new_measurement = {
            'Parameter': "4.3.2 Sea ice cover",
            'Spectral_region': MW.Spectral_region,
            'Accuracy': "Medium",
            'Id': f"{IR.Id}-syn-{MW.Id}",
            'taken_by': f"{IR.taken_by}-syn-{MW.taken_by}"
        }
        measurements.append(new_measurement)

def ocean_salinity_space_average(L, measurements):
    if L.Parameter == "3.3.1 Ocean salinity" and L.Accuracy is not None and L.Horizontal_Spatial_Resolution is not None and "SMAP_MWR" in L.taken_by and "averaged" not in L.taken_by:
        a2 = L.Accuracy / 3.0
        hsr2 = L.Horizontal_Spatial_Resolution * 3.0
        new_measurement = {
            'Parameter': L.Parameter,
            'Accuracy': a2,
            'Horizontal_Spatial_Resolution': hsr2,
            'Id': f"{L.Id}-space-averaged",
            'taken_by': f"{L.taken_by}-space-averaged"
        }
        measurements.append(new_measurement)

def ocean_wind_space_average(L, measurements):
    if L.Parameter == "3.4.1 Ocean surface wind speed" and L.Accuracy is not None and L.Horizontal_Spatial_Resolution is not None and "SMAP_MWR" in L.taken_by and "averaged" not in L.taken_by:
        a2 = L.Accuracy / 2.0
        hsr2 = L.Horizontal_Spatial_Resolution * 2.0
        new_measurement = {
            'Parameter': L.Parameter,
            'Accuracy': a2,
            'Horizontal_Spatial_Resolution': hsr2,
            'Id': f"{L.Id}-space-averaged",
            'taken_by': f"{L.taken_by}-space-averaged"
        }
        measurements.append(new_measurement)

def update_rev_time(params, orbits_used, fovs, rev_time_precomputed_index, rev_times):
    java_asserted_fact_id = 1
    rev_time_precomputed_orbit_list = ["LEO-600-polar-NA", "SSO-600-SSO-AM", "SSO-600-SSO-DD", "SSO-800-SSO-DD", "SSO-800-SSO-PM"]
    rev_time_precomputed_index = [rev_time_precomputed_orbit_list.index(orb) if orb in rev_time_precomputed_orbit_list else -1 for orb in params.getOrbitList()]

    for param in params.measurementsToInstruments.keys():
        v = r.eval("(update-fovs " + param + " (create$ " + m.stringArraytoStringWithSpaces(params.getOrbitList()) + "))")

        if RU.getTypeName(v.type()).equalsIgnoreCase("LIST"):
            thefovs = v.listValue(r.getGlobalContext())
            fovs = [str(thefovs.get(i).intValue(r.getGlobalContext())) for i in range(thefovs.size())]

            recalculate_revisit_time = any(rev_time_precomputed_index[i] == -1 for i in range(len(fovs)))

            if recalculate_revisit_time:
                coverage_granularity = 20
                lat_bounds = [math.radians(-70), math.radians(70)]
                lon_bounds = [math.radians(-180), math.radians(180)]
                field_of_view_events = []

                for orb in orbits_used:
                    fov = thefovs.get(params.getOrbitIndexes().get(orb.toString())).intValue(r.getGlobalContext())

                    if fov <= 0:
                        continue

                    field_of_view = fov
                    inclination = orb.getInclinationNum()
                    altitude = orb.getAltitudeNum()
                    raan_label = orb.getRaan()
                    num_sats = int(orb.getNum_sats_per_plane())
                    num_planes = int(orb.getNplanes())

                    accesses = coverage_analysis.getAccesses(field_of_view, inclination, altitude, num_sats, num_planes, raan_label)
                    field_of_view_events.append(accesses)

                merged_events = dict(field_of_view_events[0])

                for i in range(1, len(field_of_view_events)):
                    event = field_of_view_events[i]
                    merged_events = event_interval_merger.merge(merged_events, event, False)

                therevtimes_global = coverage_analysis.getRevisitTime(merged_events, lat_bounds, lon_bounds) / 3600
                therevtimes_us = therevtimes_global

            else:
                if len(thefovs) < 5:
                    new_fovs = [fovs[rev_time_precomputed_index[i]] for i in range(5)]
                    fovs = new_fovs

                key = "1" + " x " + m.stringArraytoStringWith(fovs, "  ")
                therevtimes_us = rev_times[key]["US"]
                therevtimes_global = rev_times[key]["Global"]

            call = "(assert (ASSIMILATION2::UPDATE-REV-TIME (parameter " + param + ") " + "(avg-revisit-time-global# " + str(therevtimes_global) + ") " + "(avg-revisit-time-US# " + str(therevtimes_us) + ")" + "(factHistory J" + str(java_asserted_fact_id) + ")))"
            java_asserted_fact_id += 1
            r.eval(call)


## aggregate_performance_score_facts

    def aggregate_performance_score_facts(params, r, m, qb):
        subobj_scores = []
        obj_scores = []
        panel_scores = []
        science = 0.0
        cost = 0.0
        fuzzy_science = None
        fuzzy_cost = None
        explanations = {}
        subobj_scores_map = {}

        try:
            vals = qb.makeQuery("AGGREGATION::VALUE")
            val = vals[0]
            science = val.getSlotValue("satisfaction").floatValue(r.getGlobalContext())
            if params.reqMode.equalsIgnoreCase("FUZZY-ATTRIBUTES") or params.reqMode.equalsIgnoreCase("FUZZY-CASES"):
                fuzzy_science = val.getSlotValue("fuzzy-value").javaObjectValue(r.getGlobalContext())
            for str_val in m.jessList2ArrayList(val.getSlotValue("sh-scores").listValue(r.getGlobalContext()), r):
                panel_scores.append(float(str_val))

            subobj_facts = qb.makeQuery("AGGREGATION::SUBOBJECTIVE")
            for f in subobj_facts:
                subobj = f.getSlotValue("id").stringValue(r.getGlobalContext())
                subobj_score = f.getSlotValue("satisfaction").floatValue(r.getGlobalContext())
                current_subobj_score = subobj_scores_map.get(subobj)
                if current_subobj_score is None or subobj_score > current_subobj_score:
                    subobj_scores_map[subobj] = subobj_score
                if subobj not in explanations:
                    explanations[subobj] = qb.makeQuery("AGGREGATION::SUBOBJECTIVE (id " + subobj + ")")

            for p in range(params.numPanels):
                nob = params.numObjectivesPerPanel[p]
                subobj_scores_p = []
                for o in range(nob):
                    subobj_p = params.subobjectives[p]
                    subobj_o = subobj_p[o]
                    nsubob = len(subobj_o)
                    subobj_scores_o = []
                    for subobj in subobj_o:
                        subobj_scores_o.append(subobj_scores_map[subobj])
                    subobj_scores_p.append(subobj_scores_o)
                subobj_scores.append(subobj_scores_p)

            for p in range(params.numPanels):
                nob = params.numObjectivesPerPanel[p]
                obj_scores_p = []
                for o in range(nob):
                    subobj_weights_p = params.subobjWeights[p]
                    subobj_weights_o = subobj_weights_p[o]
                    subobj_scores_p = subobj_scores[p]
                    subobj_scores_o = subobj_scores_p[o]
                    try:
                        obj_scores_p.append(sum([w * s for w, s in zip(subobj_weights_o, subobj_scores_o)]))
                    except Exception as e:
                        print(e)
                obj_scores.append(obj_scores_p)
        except Exception as e:
            print(e)
            traceback.print_exc()

        theresult = Result(arch, science, cost, fuzzy_science, fuzzy_cost, subobj_scores, obj_scores, panel_scores, subobj_scores_map)
        if debug:
            theresult.setCapabilities(qb.makeQuery("REQUIREMENTS::Measurement"))
            theresult.setExplanations(explanations)

        return theresult
