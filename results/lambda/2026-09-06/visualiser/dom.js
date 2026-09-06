// Small DOM/text helpers shared by the archive and GTOC12 views.

export const $ = (id) => document.getElementById(id);

export function format(value, unit = "") {
  if (value == null) return "Not applicable";
  if (!Number.isFinite(value)) return "Invalid";
  const text = value === 0 ? "0" : Math.abs(value) < 0.001 || Math.abs(value) >= 10000
    ? value.toExponential(3) : value.toFixed(4);
  return `${text}${unit ? ` ${unit}` : ""}`;
}

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[character]);
}

export function metricRows(rows) {
  return rows.map(([name, value]) =>
    `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}
