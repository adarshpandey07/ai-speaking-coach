/**
 * Visual-only progress indicator for the MVP: which week/day the student is on
 * and their streak. Nothing here drives behaviour.
 */
export default function ProgressBadge({ week, day, streak, topic, mode }) {
  if (mode === 'baseline') {
    return (
      <div className="badge badge-baseline">
        <span className="badge-title">First session</span>
        <span className="badge-sub">Let&apos;s hear how you speak today</span>
      </div>
    )
  }
  return (
    <div className="badge">
      <span className="badge-title">
        Week {week} · Day {day}
      </span>
      <span className="badge-sub">{topic}</span>
      <span className="badge-streak" title={`${streak} day streak`}>
        🔥 {streak}
      </span>
    </div>
  )
}
