import type { ReactNode } from 'react';
import { Activity, Info } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ADULT_COLOR, NON_ADULT_COLOR, count, pct, METRIC_NAMES, runColor } from './data';
import type { Client, Metric, Metrics, Run, Split } from './types';

export function Panel({title, subtitle, action, children, className = ''}: { title: string; subtitle?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><header className="panel-header"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</header>{children}</section>;
}
export function Empty({title = 'No measurements available', children}: {title?: string; children: ReactNode}) {
  return <div className="empty"><Activity size={25}/><h3>{title}</h3><p>{children}</p></div>;
}
export function Note({children}: {children: ReactNode}) { return <div className="note"><Info size={15}/><span>{children}</span></div>; }
export function Kpi({label, value, detail, accent = false}: {label: string; value: string; detail?: string; accent?: boolean}) {
  return <div className={`kpi ${accent ? 'accent' : ''}`}><span className="kpi-label">{label}</span><strong className={value.includes('Not ') ? 'missing' : ''}>{value}</strong>{detail && <span className="kpi-detail">{detail}</span>}</div>;
}
export function DataBadge({kind}: {kind: string}) { return <span className={`badge ${kind === 'real' ? 'green' : 'amber'}`}>{kind === 'synthetic' ? 'Synthetic smoke' : kind === 'real' ? 'Real data' : 'Unknown data'}</span>; }
const grid = '#253044';
const tooltip = { background: '#182231', border: '1px solid #35425b', borderRadius: 9, color: '#edf2fb', fontSize: 12 };

export function TrendChart({runs, metric, split, range}: {runs: Run[]; metric: Metric; split: Split; range: [number, number]}) {
  const numbers = [...new Set(runs.flatMap(run => run.rounds.map(r => r.round_number)))].sort((a,b) => a-b).filter(n => n >= range[0] && n <= range[1]);
  const rows = numbers.map(round => Object.fromEntries([['round', round], ...runs.map(run => [run.experiment_id,
    run.rounds.find(r => r.round_number === round)?.metrics[split]?.[metric] ?? null])]));
  const available = runs.some(run => run.rounds.some(r => r.round_number >= range[0] && r.round_number <= range[1] && r.metrics[split]?.[metric] != null));
  if (!available) return <Empty title="This metric has not been measured">Choose a different metric, split, or round range. Test data appears only at a recorded evaluation checkpoint.</Empty>;
  return <div className="chart tall" role="img" aria-label={`${METRIC_NAMES[metric]} by federated round. Exact values in experiment history and CSV export.`}><ResponsiveContainer width="100%" height="100%">
    <LineChart data={rows} margin={{top: 12, right: 18, left: 0, bottom: 6}} accessibilityLayer>
      <CartesianGrid stroke={grid} strokeDasharray="3 5" vertical={false}/>
      <XAxis dataKey="round" type="number" domain={['dataMin', 'dataMax']} allowDecimals={false} tick={{fill: '#8e9bb0', fontSize: 11}} tickLine={false} axisLine={false} label={{ value: 'Federated round', position: 'insideBottomRight', offset: -5, fill: '#8e9bb0', fontSize: 11 }}/>
      <YAxis domain={metric === 'loss' ? [0, 'auto'] : [0, 1]} tickFormatter={v => metric === 'loss' ? Number(v).toFixed(1) : `${Math.round(Number(v)*100)}%`} tick={{fill: '#8e9bb0', fontSize: 11}} axisLine={false} tickLine={false} width={48}/>
      <Tooltip contentStyle={tooltip} formatter={value => value == null ? 'Not measured' : metric === 'loss' ? Number(value).toFixed(4) : pct(Number(value))} labelFormatter={v => `Round ${v}`}/>
      {runs.map((run, i) => <Line key={run.experiment_id} name={run.name} dataKey={run.experiment_id} type="linear" stroke={runColor(run.experiment_id)} strokeWidth={2.3} strokeDasharray={i % 2 ? '6 4' : undefined} dot={{r: 3.5}} activeDot={{r: 6}} connectNulls={false} isAnimationActive={false}/>)}
    </LineChart>
  </ResponsiveContainer></div>;
}

export function ClassChart({clients, after = false}: {clients: Client[]; after?: boolean}) {
  if (!clients.length) return <Empty>No client update counts are available for this round.</Empty>;
  const rows = clients.map(c => ({client: `Client ${c.client_id}`, Adult: after ? c.used_adult : c.original_adult, 'Non-adult': after ? c.used_non_adult : c.original_non_adult}));
  return <div className="chart" role="img" aria-label={`${after ? 'Balanced' : 'Original'} training class counts by client`}><ResponsiveContainer width="100%" height="100%"><BarChart data={rows} barGap={4} margin={{top: 10, right: 15, left: 0, bottom: 0}} accessibilityLayer>
    <CartesianGrid stroke={grid} strokeDasharray="3 5" vertical={false}/><XAxis dataKey="client" tick={{fill:'#9ba8bd', fontSize:11}} tickLine={false} axisLine={false}/>
    <YAxis tickFormatter={v => Number(v).toLocaleString('en-US', {notation:'compact'})} tick={{fill:'#9ba8bd', fontSize:11}} tickLine={false} axisLine={false} width={48}/>
    <Tooltip contentStyle={tooltip} formatter={v => count(Number(v))}/><Legend wrapperStyle={{fontSize:12, paddingTop:12}}/>
    <Bar dataKey="Adult" fill={ADULT_COLOR} radius={[3,3,0,0]} maxBarSize={30} isAnimationActive={false}/><Bar dataKey="Non-adult" fill={NON_ADULT_COLOR} radius={[3,3,0,0]} maxBarSize={30} isAnimationActive={false}/>
  </BarChart></ResponsiveContainer></div>;
}

export function ConfusionMatrix({metrics}: {metrics?: Metrics}) {
  if (!metrics) return <Empty title="No confusion matrix for this split">The matrix appears when this split is evaluated. No prediction counts are inferred.</Empty>;
  return <div className="matrix-wrap"><div className="matrix-predicted">Predicted class</div><div className="matrix"><span className="matrix-axis">True class</span><span/><span className="matrix-heading">Non-adult</span><span className="matrix-heading">Adult</span>
    {metrics.confusion_matrix.map((row, actual) => <div className="matrix-row" key={actual}><span className="matrix-heading">{actual ? 'Adult' : 'Non-adult'}</span>{row.map((v, predicted) => {
      const support = row.reduce((a,b) => a+b,0); const rate = support ? v/support : 0;
      return <div key={predicted} className={`matrix-cell ${actual === predicted ? 'correct' : 'incorrect'}`} style={{'--strength': rate} as React.CSSProperties} title={`${actual ? 'Adult' : 'Non-adult'} classified as ${predicted ? 'Adult' : 'Non-adult'}: ${count(v)}`}><small>{[['TN','FP'],['FN','TP']][actual][predicted]}</small><strong>{count(v)}</strong><span>{pct(rate)} of true class</span></div>;
    })}</div>)}
  </div><p className="chart-caption">N = {count(metrics.n_records)} · Rows: actual labels · Columns: predictions</p></div>;
}

export function MetricBars({rows, metric}: {rows: {name: string; value: number | null; color: string}[]; metric: string}) {
  return <div className="metric-bars">{rows.map((row, i) => <div className="metric-bar" key={i}><div><span>{row.name}</span><strong>{row.value == null ? 'Not measured' : metric === 'duration' ? `${row.value.toFixed(2)} s` : pct(row.value)}</strong></div><div className="bar-track"><span style={{width: `${row.value == null ? 0 : metric === 'duration' ? 100*row.value/Math.max(1,...rows.map(r => r.value ?? 0)) : 100*row.value}%`, background:row.color}}/></div></div>)}</div>;
}
