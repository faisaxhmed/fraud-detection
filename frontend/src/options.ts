/** Dropdown option sources, copied from backend/models/*.json at build time. */
import categoryOptionsRaw from './data/category_options.json';
import cityPopulationLookupRaw from './data/city_population_lookup.json';

export const categoryOptions = categoryOptionsRaw as {
  merchant: string[];
  category: string[];
  job: string[];
};

/** Display label for each raw category code — the raw value is still what's sent to the backend. */
export const CATEGORY_LABELS: Record<string, string> = {
  grocery_pos: 'Grocery (in-person)',
  grocery_net: 'Grocery (online)',
  gas_transport: 'Gas & Transport',
  misc_pos: 'Miscellaneous (in-person)',
  misc_net: 'Miscellaneous (online)',
  shopping_pos: 'Shopping (in-person)',
  shopping_net: 'Shopping (online)',
  food_dining: 'Food & Dining',
  health_fitness: 'Health & Fitness',
  kids_pets: 'Kids & Pets',
  personal_care: 'Personal Care',
  home: 'Home',
  entertainment: 'Entertainment',
  travel: 'Travel',
};

/** Single source of truth for field display names, shared by the form and the SHAP chart so
 * the same feature reads identically everywhere (e.g. never "Hour of Day" in one place and
 * "Trans Hour" in the other). */
export const FEATURE_LABELS: Record<string, string> = {
  amount: 'Amount',
  merchant: 'Merchant',
  category: 'Category',
  gender: 'Gender',
  city_pop: 'City Population',
  job: 'Job',
  trans_hour: 'Hour of Day',
  trans_dayofweek: 'Day of Week',
  age: 'Age',
  distance: 'Distance',
};

const cityPopulationLookup = cityPopulationLookupRaw as Record<string, number>;

export interface CityStateOption {
  key: string;
  city: string;
  state: string;
  label: string;
}

/** "City, State" options derived from the lookup table's keys, sorted for a searchable select. */
export const cityStateOptions: CityStateOption[] = Object.keys(cityPopulationLookup)
  .map((key) => {
    const [city, state] = key.split('|');
    return { key, city, state, label: `${city}, ${state}` };
  })
  .sort((a, b) => a.label.localeCompare(b.label));
