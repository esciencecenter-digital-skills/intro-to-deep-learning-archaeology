# Import the libraries
import pandas as pd
import numpy as np

# Download the datase here > https://belowthesurface.amsterdam/en/pagina/publicaties-en-datasets

# Load the data from the csv file

df = pd.read_csv('C://data//rokin//Downloadtabel_EN.csv')

# The fields have been renamed from Dutch to English, although we do not use all the fields in the workshop we kept all the translations here.
df = df.rename(columns={
    "vondstnummer": "find_number",
    "project_code": "project_code_location",
    "categorie": "material_category",
    "subcategorie": "material",
    "gewicht": "weight_in_grams",
    "object": "object_name",
    "objectdeel": "object_part",
    "fragmenten": "number_of_fragments",
    "past_aan_hoort_bij": "fits_or_belongs_to_find_numbers",
    "begin_dat": "start_date",
    "eind_dat": "end_date",
    "put": "trench_number",
    "vlak": "level_number",
    "spoor": "feature_number",
    "vak": "section_number",
    "vlak_min": "minimum_level_height",
    "vlak_max": "maximum_level_height",
    "trefwoorden": "keywords_photographed_finds_only",
    "niveau1": "level_1_of_the_functional_classification",
    "niveau2": "level_2_of_the_functional_classification",
    "niveau3": "level_3_of_the_functional_classification",
    "niveau4": "level_4_of_the_functional_classification",
    "website": "on_website",
    "aardewerk_ds_type": "ceramics_deventer_system_code",
    "aardewerk_eve_rand": "ceramics_rim_eve_estimated_vessel_equivalent",
    "aardewerk_eve_bodem": "ceramics_base_eve_estimated_vessel_equivalent",
    "aardewerk_herkomst": "ceramics_location_of_production",
    "aardewerk_object_diameter_mm": "ceramics_reconstructed_object_diameter_in_mm",
    "aardewerk_object_hoogte_mm": "ceramics_reconstructed_object_height_in_mm",
    "aardewerk_fragm_lengte_mm": "ceramics_fragment_length_in_mm",
    "aardewerk_fragm_breedte_mm": "ceramics_fragment_width_in_mm",
    "aardewerk_fragm_hoogte_mm": "ceramics_fragment_height_in_mm",
    "aardewerk_opp_behandeling": "ceramics_surface_treatment",
    "aardewerk_decoratietechniek": "ceramics_decoration_technique",
    "aardewerk_decorgroepen": "ceramics_image_type",
    "aardewerk_merk": "ceramics_mark",
    "blankewapens_totale_lengte_mm": "hilted_weapons_total_length_in_mm",
    "blankewapens_type_lemmet": "hilted_weapons_blade_type",
    "blankewapens_type_klingvanger": "hilted_weapons_blade_catcher_type",
    "blankewapens_type_heft": "hilted_weapons_grip_type",
    "blankewapens_materiaal_heft": "hilted_weapons_grip_material",
    "blankewapens_type_heftbekroning": "hilted_weapons_pommel_cap_type",
    "blankewapens_productiemerk": "hilted_weapons_production_mark",
    "bouwmaterialen_kleur": "building_material_colour",
    "bouwmaterialen_grootstelengte_mm": "building_material_greatest_length_in_mm",
    "bouwmaterialen_grootstebreedte_mm": "building_material_greatest_width_in_mm",
    "bouwmaterialen_grootstedikte_mm": "building_material_greatest_thickness_in_mm",
    "bouwmaterialen_productiecentrum": "building_material_production_centre",
    "bouwmaterialen_afbeelding": "building_material_image_type",
    "bouwmaterialen_oppervlakte": "building_material_surface_treatment",
    "fauna_soort": "fauna_species",
    "fauna_element": "fauna_element",
    "fauna_lengte_mm": "fauna_length_in_mm",
    "fauna_breedte_mm": "fauna_width_in_mm",
    "fauna_hoogte_mm": "fauna_height_in_mm",
    "glas_ds_type": "glass_deventer_system_code",
    "glas_kleur": "glass_colour",
    "glas_herkomst": "glass_location_of_production",
    "glas_eve_rand": "glass_rim_eve_estimated_vessel_equivalent",
    "glas_eve_bodem": "glass_base_eve_estimated_vessel_equivalent",
    "glas_object_diameter_mm": "glass_reconstructed_object_diameter_in_mm",
    "glas_object_hoogte_mm": "glass_reconstructed_object_height_in_mm",
    "glas_fragm_lengte_mm": "glass_fragment_length_in_mm",
    "glas_fragm_breedte_mm": "glass_fragment_width_in_mm",
    "glas_fragm_hoogte_mm": "glass_fragment_height_in_mm",
    "glas_fragm_dikte_mm": "glass_fragment_thickness_in_mm",
    "glas_decoratie": "glass_decoration",
    "glas_merk": "glass_mark",
    "hout_deelmaterialen": "wood_secondary_material",
    "hout_grootstelengte_mm": "wood_greatest_length_in_mm",
    "hout_grootstebreedte_mm": "wood_greatest_width_in_mm",
    "hout_grootstehoogte_mm": "wood_greatest_height_in_mm",
    "hout_grootstedikte_mm": "wood_greatest_thickness_in_mm",
    "hout_diameter_mm": "wood_diameter_in_mm",
    "hout_productiewijze": "wood_production_method",
    "kunststof_deelmaterialen": "synthetics_secondary_material",
    "kunststof_grootstelengte_mm": "synthetics_greatest_length_in_mm",
    "kunststof_grootstebreedte_mm": "synthetics_greatest_width_in_mm",
    "kunststof_grootstehoogte_mm": "synthetics_greatest_height_in_mm",
    "kunststof_diameter_mm": "synthetics_diameter_in_mm",
    "kunststof_productiecentrum": "synthetics_production_centre",
    "kunststof_eigenaar": "synthetics_owner",
    "kunststof_merk": "synthetics_mark",
    "kunststof_type": "synthetics_type",
    "kunststof_eenheid_waarde": "synthetics_unit_value",
    "leer_archeologischobjecttype": "leather_type",
    "leer_deelmaterialen": "leather_secondary_material",
    "leer_versiering": "leather_decoration_technique",
    "leer_leersoort": "leather_leather_type",
    "leer_grootste_lengte_mm": "leather_greatest_length_in_mm",
    "leer_grootste_breedte_mm": "leather_greatest_width_in_mm",
    "leer_grootste_hoogte_mm": "leather_greatest_height_in_mm",
    "messen_angel_of_plaatangel": "knives_whittle_tang_or_scale_tang",
    "messen_heft_mat_1": "knives_hilt_material",
    "messen_lemmet_lengte": "knives_blade_length_in_mm",
    "messen_heft_lengte": "knives_hilt_length_in_mm",
    "messen_minimale_totale_lengte": "knives_greatest_length",
    "messen_minimale_totale_breedte": "knives_greatest_width",
    "metaal_deelmaterialen": "metal_secondary_material",
    "metaal_grootstelengte_mm": "metal_greatest_length_in_mm",
    "metaal_grootstebreedte_mm": "metal_greatest_width_in_mm",
    "metaal_grootstehoogte_mm": "metal_greatest_height_in_mm",
    "metaal_grootstedikte_mm": "metal_greatest_thickness_in_mm",
    "metaal_diameter_mm": "metal_diameter_in_mm",
    "metaal_productiecentrum": "metal_production_centre",
    "mix_grootstelengte_mm": "mixed_greatest_length_in_mm",
    "mix_grootstebreedte_mm": "mixed_greatest_width_in_mm",
    "mix_grootstehoogte_mm": "mixed_greatest_height_in_mm",
    "mix_grootstedikte_mm": "mixed_greatest_thickness_in_mm",
    "mix_diameter_mm": "mixed_diameter_in_mm",
    "munt_land_geografisch": "coins_country_geographical",
    "munt_staat_politieke_eenheid": "coins_political_entity",
    "munt_autoriteit_politiek": "coins_authority",
    "munt_muntsoort": "coins_coin_type",
    "munt_eenheid_waarde": "coins_denomination_value",
    "munt_ontwerper": "coins_designer",
    "munt_onderwerp_gelegenheid": "coins_subject_or_occasion",
    "munt_muntplaats_productieplaats": "coins_location_of_production",
    "munt_lengte_diameter_in_mm": "coins_diameter_in_mm",
    "munt_breedte_in_mm": "coins_width_in_mm",
    "natuursteen_subsoort": "natural_stone_sub_type",
    "natuursteen_grootstelengte_mm": "natural_stone_greatest_length_in_mm",
    "natuursteen_grootstebreedte_mm": "natural_stone_greatest_width_in_mm",
    "natuursteen_grootstehoogte_mm": "natural_stone_greatest_height_in_mm",
    "natuursteen_grootstedikte_mm": "natural_stone_greatest_thickness_in_mm",
    "natuursteen_diameter_mm": "natural_stone_diameter_in_mm",
    "natuursteen_productiesporen": "natural_stone_production_marks",
    "plant_soort": "botanical_species",
    "plant_grootstelengte_mm": "botanical_greatest_length_in_mm",
    "plant_grootstebreedte_mm": "botanical_greatest_width_in_mm",
    "plant_grootstehoogte_mm": "botanical_greatest_height_in_mm",
    "plant_grootstedikte_mm": "botanical_greatest_thickness_in_mm",
    "plant_diameter_mm": "botanical_diameter_in_mm",
    "rookpijpen_model": "pipes_model",
    "rookpijpen_zijmerk_links": "pipes_mark_on_side_of_bowl_left",
    "rookpijpen_zijmerk_rechts": "pipes_mark_on_side_of_bowl_right",
    "rookpijpen_bijmerk_links": "pipes_mark_on_base_of_heel_left",
    "rookpijpen_bijmerk_rechts": "pipes_mark_on_base_of_heel_right",
    "rookpijpen_merk_of_hielmerk": "pipes_mark_or_mark_on_base_of_heel",
    "rookpijpen_oppervlaktebehandeling_kop": "pipes_surface_treatment_bowl",
    "rookpijpen_kopopening": "pipes_bowl_opening",
    "rookpijpen_radering": "pipes_milling",
    "rookpijpen_steelbehandeling": "pipes_stem_treatment",
    "rookpijpen_kwaliteit": "pipes_quality",
    "rookpijpen_productiecentrum": "pipes_production_centre",
    "rookpijpen_pijpenmaker": "pipes_pipe_maker",
    "sculpturen_baksel": "sculptures_fabric",
    "sculpturen_hoogte_mm": "sculptures_height_in_mm",
    "sculpturen_breedte_mm": "sculptures_width_in_mm",
    "sculpturen_diepte_mm": "sculptures_depth_in_mm",
    "sculpturen_hol_of_massief": "sculptures_hollow_or_solid",
    "sculpturen_voorstelling": "sculptures_image_description",
    "textiel_deelmaterialen": "textile_secondary_material",
    "textiel_grootstelengte_mm": "textile_greatest_length_in_mm",
    "textiel_grootstebreedte_mm": "textile_greatest_width_in_mm",
    "textiel_productiewijze": "textile_production_method",
    "textiel_binding": "textile_binding",
    "textiel_bewerking": "textile_processing",
    "touw_grootstelengte_mm": "rope_greatest_length_in_mm",
    "touw_grootstebreedte_mm": "rope_greatest_width_in_mm",
    "touw_diameter_mm": "rope_diameter_in_mm",
    "touw_strengen": "rope_number_of_strands",
    "touw_productiewijze": "rope_production_method"})
	
# create a subset for the ceremacs based on the specified conditions
subset = df[(df['material_category'] == 'CER') & 
                    (~df['ceramics_reconstructed_object_height_in_mm'].isnull()) &
                    (~df['ceramics_reconstructed_object_diameter_in_mm'].isnull()) &
                    (~df['level_1_of_the_functional_classification'].isnull()) &
                    (~df['level_2_of_the_functional_classification'].isnull()) &
                    (~df['start_date'].isnull()) &
                    (~df['end_date'].isnull()) &
                    (df['level_1_of_the_functional_classification'] == 'Food Processing & Consumption') &
                    (~df['level_2_of_the_functional_classification'].isin(['Supplies, food general', 'Supplies, liquids general']))
                   ]
				   
# Subset of the dataset is created 
rokin_cer = subset[['find_number', 'material', 'start_date', 'end_date', 
                    'level_2_of_the_functional_classification',
                    'ceramics_reconstructed_object_diameter_in_mm', 'ceramics_reconstructed_object_height_in_mm', 
                    'ceramics_image_type','ceramics_mark', 'on_website']]

# Convert doubles to integers (for dates)
rokin_cer['start_date'] = rokin_cer['start_date'].astype(int)
rokin_cer['end_date'] = rokin_cer['end_date'].astype(int)

# material categories are merged
rokin_cer.replace('faience','faience', inplace=True)
rokin_cer.replace('faience: French','faience', inplace=True)
rokin_cer.replace('faience: Holland','faience', inplace=True)
rokin_cer.replace('faience: Italian','faience', inplace=True)
rokin_cer.replace('gold lustre','gold lustre', inplace=True)
rokin_cer.replace('greyware: hand-built','greyware', inplace=True)
rokin_cer.replace('industrial ware: black','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: coloured','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: creamware','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: pearlware','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: red','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: rewdware','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: scratchware','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: stoneware','industrial ware', inplace=True)
rokin_cer.replace('industrial ware: white','industrial ware', inplace=True)
rokin_cer.replace('maiolica','maiolica', inplace=True)
rokin_cer.replace('maiolica: Italian','maiolica', inplace=True)
rokin_cer.replace('maiolica: Spanish','maiolica', inplace=True)
rokin_cer.replace('porcelain: capucin','porcelain', inplace=True)
rokin_cer.replace('porcelain: China','porcelain', inplace=True)
rokin_cer.replace('porcelain: Europe','porcelain', inplace=True)
rokin_cer.replace('porcelain: famille rose','porcelain', inplace=True)
rokin_cer.replace('porcelain: famille verte','porcelain', inplace=True)
rokin_cer.replace('porcelain: Germany','porcelain', inplace=True)
rokin_cer.replace('porcelain: Japan','porcelain', inplace=True)
rokin_cer.replace('porcelain: KhangXi','porcelain', inplace=True)
rokin_cer.replace('porcelain: transitional','porcelain', inplace=True)
rokin_cer.replace('porcelain: WanLi','porcelain', inplace=True)
rokin_cer.replace('redware','redware', inplace=True)
rokin_cer.replace('redware, slip','redware', inplace=True)
rokin_cer.replace('redware, slip: Lower Rhine region','redware', inplace=True)
rokin_cer.replace('redware, slip: sgraffito','redware', inplace=True)
rokin_cer.replace('redware, slip: slip-cup decoration','redware', inplace=True)
rokin_cer.replace('redware, slip: Werra','redware', inplace=True)
rokin_cer.replace('redware: Frankfurt tradition','redware', inplace=True)
rokin_cer.replace('redware: French','redware', inplace=True)
rokin_cer.replace('redware: Iberian','redware', inplace=True)
rokin_cer.replace('redware: slip applied by brush','redware', inplace=True)
rokin_cer.replace('redware: unglazed','redware', inplace=True)
rokin_cer.replace('redware: western Germany','redware', inplace=True)
rokin_cer.replace('stoneware','stoneware', inplace=True)
rokin_cer.replace('stoneware: Aachen','stoneware', inplace=True)
rokin_cer.replace('stoneware: Cologne','stoneware', inplace=True)
rokin_cer.replace('stoneware: Frechen','stoneware', inplace=True)
rokin_cer.replace('stoneware: Langerwehe','stoneware', inplace=True)
rokin_cer.replace('stoneware: Raeren','stoneware', inplace=True)
rokin_cer.replace('stoneware: Siegburg','stoneware', inplace=True)
rokin_cer.replace('stoneware: Westerwald','stoneware', inplace=True)
rokin_cer.replace('whiteware','whiteware', inplace=True)
rokin_cer.replace('whiteware: Frankfurt tradition','whiteware', inplace=True)
rokin_cer.replace('whiteware: Germany','whiteware', inplace=True)


# To avoid that the participants need to write long strings we have simplified the collumn names a little
rokin_cer.rename(columns={
    "level_2_of_the_functional_classification": "l2_class",
    "ceramics_reconstructed_object_diameter_in_mm": "object_diameter",
    "ceramics_reconstructed_object_height_in_mm": "object_height"})

# For artefacts of which a picture is available at the website of below the surface we added a link 
rokin_cer['url'] = np.where(rokin_cer['on_website'] == 1, ('https://belowthesurface.amsterdam/en/vondst/'+rokin_cer['find_number']),"")

# Here we store the file.
rokin_cer.to_csv('C://data//rokin//rokin_cer.csv', index=False)