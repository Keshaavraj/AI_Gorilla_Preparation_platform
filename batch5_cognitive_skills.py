"""
Batch 5: Cognitive Skills Questions
- Problem Solving (15 questions)
- Critical Thinking (15 questions)
"""

from database_manager import QuestionDatabase, create_question

def populate_problem_solving():
    """15 Problem Solving Questions"""
    questions = [
        create_question(
            "A data pipeline is failing intermittently. What is the BEST first step to diagnose the issue?",
            [
                "Immediately rewrite the entire pipeline",
                "Gather logs and identify patterns in when failures occur",
                "Restart the server repeatedly",
                "Assume it's a hardware issue"
            ],
            1,
            "Systematic problem-solving starts with data collection and pattern analysis. Logs reveal: failure frequency, error messages, resource usage, timing patterns. This evidence guides hypothesis formation. Common patterns: time-based (scheduled jobs), data-based (specific inputs), resource-based (memory/CPU). Premature solutions (rewriting code) waste time. The scientific method applies: observe, hypothesize, test.",
            "Medium",
            85
        ),
        create_question(
            "Your ML model performs well in development but poorly in production. What is the MOST likely cause?",
            [
                "The model architecture is wrong",
                "Train-test distribution mismatch (data drift)",
                "The programming language is wrong",
                "Too much training data"
            ],
            1,
            "Production data often differs from training data (concept drift, data drift, covariate shift). Causes: time-based changes, different user populations, data collection biases. Solutions: (1) monitor data distributions, (2) retrain regularly, (3) use validation data closer to production, (4) A/B test carefully. Always validate on realistic data. This is more common than architectural issues when dev performance is good.",
            "Hard",
            95
        ),
        create_question(
            "You need to optimize a slow SQL query. What should you check FIRST?",
            [
                "Rewrite in a different language",
                "Check if indexes exist on columns used in WHERE, JOIN, and ORDER BY clauses",
                "Buy faster hardware",
                "Remove all data"
            ],
            1,
            "Missing indexes are the most common cause of slow queries. Indexes enable fast lookups (B-tree traversal O(log n) vs table scan O(n)). Check: (1) WHERE clause columns, (2) JOIN columns, (3) ORDER BY columns. Use EXPLAIN/EXPLAIN ANALYZE to see query plan. Other issues: SELECT *, unnecessary JOINs, N+1 queries. But indexes give biggest wins with least effort. Beware: too many indexes slow writes.",
            "Medium",
            90
        ),
        create_question(
            "A user reports a bug you cannot reproduce. What is the BEST approach?",
            [
                "Tell them the bug doesn't exist",
                "Gather detailed information: environment, steps to reproduce, error messages, screenshots",
                "Close the ticket immediately",
                "Guess randomly"
            ],
            1,
            "Unreproducible bugs require systematic information gathering: (1) exact steps taken, (2) environment (OS, browser, version), (3) error messages/logs, (4) screenshots/video, (5) data state. Often bugs are environment-specific (browser compatibility), timing-dependent (race conditions), or data-dependent (edge cases). Create a questionnaire to gather this systematically. Never assume bug doesn't exist - 'works on my machine' is a red flag.",
            "Medium",
            80
        ),
        create_question(
            "Your API response time degraded from 100ms to 2000ms after a deployment. How do you identify the cause?",
            [
                "Roll back immediately without investigation",
                "Profile the new code, check database query times, examine external API calls, review recent changes",
                "Blame the database team",
                "Wait for it to fix itself"
            ],
            1,
            "Performance regression requires profiling: (1) application profiler (identify slow functions), (2) database query analyzer (slow queries), (3) APM tools (external calls), (4) code diff review. Common culprits: N+1 queries, inefficient algorithms, missing caching, external API timeouts. Measure, don't guess. Tools: cProfile (Python), flame graphs, database slow query logs. Compare before/after metrics. Rollback may be necessary, but understanding prevents recurrence.",
            "Hard",
            95
        ),
        create_question(
            "You're asked to reduce cloud costs by 30%. What's the MOST effective approach?",
            [
                "Delete all data",
                "Analyze resource utilization metrics, identify over-provisioned resources, and implement autoscaling",
                "Switch cloud providers randomly",
                "Shut down everything"
            ],
            1,
            "Data-driven cost optimization: (1) identify largest cost centers (analytics dashboard), (2) find waste (idle resources, over-provisioning), (3) right-size instances, (4) use reserved/spot instances, (5) implement autoscaling, (6) archive old data to cheaper storage. Common wins: unused dev environments, over-provisioned databases, lack of autoscaling. 80/20 rule: 20% of resources likely account for 80% of costs.",
            "Hard",
            90
        ),
        create_question(
            "Two features conflict in requirements. Feature A needs low latency, Feature B needs high throughput. How do you proceed?",
            [
                "Choose randomly",
                "Understand business priorities, measure trade-offs quantitatively, propose compromise or separate optimized paths",
                "Implement neither",
                "Argue with stakeholders"
            ],
            1,
            "Conflicting requirements need: (1) stakeholder alignment on priorities (which business need is more critical?), (2) quantify trade-offs (latency vs throughput numbers), (3) explore solutions (can we have separate endpoints? different tiers?), (4) propose data-driven recommendation. Often false dichotomy - creative solutions exist. Example: async processing for throughput, synchronous for latency. Document decision rationale.",
            "Hard",
            95
        ),
        create_question(
            "Your team's velocity has dropped 50%. What should you investigate?",
            [
                "Fire everyone",
                "Check for blockers, technical debt, context switching, unclear requirements, team morale",
                "Ignore the metrics",
                "Work longer hours only"
            ],
            1,
            "Velocity drops indicate systemic issues: (1) technical debt slowing development, (2) unclear/changing requirements causing rework, (3) blockers (waiting for reviews, deployments, dependencies), (4) context switching (too many projects), (5) team issues (morale, turnover). Talk to the team - they know the blockers. Measure: time in code review, deployment frequency, rework percentage. Address root causes, not symptoms. Working longer hours treats symptom, not cause.",
            "Medium",
            85
        ),
        create_question(
            "You discover a security vulnerability in production code. What's the correct sequence of actions?",
            [
                "Post about it on social media",
                "Assess severity, develop fix, deploy patch, review how it was introduced, prevent recurrence",
                "Do nothing",
                "Blame the intern"
            ],
            1,
            "Security incident response: (1) assess severity (data exposure? active exploitation?), (2) contain (if needed, take system offline), (3) develop and test fix, (4) deploy urgently, (5) post-mortem (how introduced? why not caught?), (6) prevent recurrence (automated security scanning, training). Document timeline. Notify affected users if data compromised. Learn, don't blame. Security is everyone's responsibility.",
            "Medium",
            90
        ),
        create_question(
            "You're assigned a vague requirement: 'Make the system faster.' What do you do?",
            [
                "Start coding randomly",
                "Define metrics, measure current performance, identify bottlenecks, set specific targets with stakeholders",
                "Quit",
                "Say it's already fast enough"
            ],
            1,
            "Vague requirements need clarification: (1) What is 'faster'? (latency? throughput? page load?), (2) Measure current state (baseline), (3) Set specific targets (e.g., 'reduce p95 latency from 500ms to 200ms'), (4) Identify bottlenecks (profile), (5) Get stakeholder agreement on priorities. Without metrics, cannot measure success. Always define done. Specific, measurable goals enable focused optimization and validation.",
            "Hard",
            95
        ),
        create_question(
            "Your test suite takes 2 hours to run. How do you improve this?",
            [
                "Delete all tests",
                "Parallelize tests, identify slow tests, use test categorization (unit/integration), optimize fixtures",
                "Never run tests again",
                "Buy faster computer only"
            ],
            1,
            "Slow test suites hurt productivity. Solutions: (1) parallelize (run tests concurrently), (2) profile tests (find slow ones), (3) categorize (unit tests fast/always, integration slower/less frequent), (4) optimize setup/teardown, (5) mock external dependencies, (6) selective running (test only affected code). Goal: <10 min for most runs, comprehensive suite nightly. Fast feedback loop crucial for TDD and CI/CD.",
            "Medium",
            85
        ),
        create_question(
            "A stakeholder wants a feature that will take 3 months but provides minimal value. How do you respond?",
            [
                "Immediately start building",
                "Understand the underlying need, propose simpler alternatives, discuss cost-benefit with data",
                "Refuse without explanation",
                "Build it poorly intentionally"
            ],
            1,
            "Challenge requirements productively: (1) understand the why (underlying business need), (2) propose alternatives (simpler solutions to same need?), (3) quantify cost (3 months = opportunity cost of other features), (4) quantify value (how many users? revenue impact?), (5) collaborative decision with data. Sometimes seemingly simple requests mask complex needs. Other times, 80% of value achievable with 20% effort. Build trust through questioning, not defiance.",
            "Hard",
            95
        ),
        create_question(
            "You inherit legacy code with no documentation. What's the BEST way to understand it?",
            [
                "Delete it and start over",
                "Read code systematically, trace execution with debugger, write tests to verify behavior, document as you learn",
                "Never touch it",
                "Complain constantly"
            ],
            1,
            "Understanding legacy code: (1) start with high-level architecture (what are main components?), (2) trace key workflows with debugger, (3) write characterization tests (document current behavior), (4) identify patterns and idioms, (5) document as you learn. Don't rewrite unless necessary - rewrites rarely succeed and lose institutional knowledge. Tests provide safety net for changes. Refactor incrementally with test coverage.",
            "Medium",
            90
        ),
        create_question(
            "Your monitoring shows disk usage at 95%. What should you do FIRST?",
            [
                "Delete random files",
                "Identify what's consuming space (logs? temp files? data growth?), then take appropriate action",
                "Ignore it until 100%",
                "Buy more servers"
            ],
            1,
            "Disk space crisis requires quick assessment: (1) identify space consumers (du, df commands), (2) check for unexpected growth (logs exploding? temp files not cleaned?), (3) immediate relief (compress/delete old logs), (4) long-term solution (log rotation, data archival, storage expansion). Common culprits: unrotated logs, temp files, abandoned data. Set up alerts at 80% to act before emergency. Automate cleanup where safe.",
            "Medium",
            80
        ),
        create_question(
            "You need to estimate a complex project with many unknowns. What approach should you take?",
            [
                "Pick a random number",
                "Break into smaller tasks, estimate ranges (best/likely/worst), identify risks and unknowns, add buffer",
                "Always say 1 week",
                "Let someone else decide"
            ],
            1,
            "Estimation with uncertainty: (1) decompose into smaller, more estimable tasks, (2) use three-point estimates (optimistic/likely/pessimistic), (3) identify dependencies and risks, (4) aggregate with uncertainty (Monte Carlo), (5) communicate confidence levels. Unknowns require discovery time (spikes). Past data helps. Cone of uncertainty narrows over time. Better to give ranges with confidence than false precision. Re-estimate as you learn.",
            "Hard",
            95
        )
    ]
    return questions


def populate_critical_thinking():
    """15 Critical Thinking Questions"""
    questions = [
        create_question(
            "A colleague claims 'Our new ML model is 95% accurate, so it's ready for production.' What's the issue with this reasoning?",
            [
                "95% is always good enough",
                "Accuracy alone is insufficient - need precision, recall, and understanding of class distribution",
                "The model should be 100% accurate",
                "Accuracy is irrelevant"
            ],
            1,
            "Accuracy is misleading especially with imbalanced classes. For 95% negative class, always predicting negative gives 95% accuracy but is useless. Need: (1) precision/recall/F1, (2) confusion matrix, (3) per-class metrics, (4) business impact of errors (FP vs FN cost). Critical thinking: question single metrics, consider context. Production readiness requires: performance on relevant metrics, robustness, monitoring, rollback plan.",
            "Hard",
            95
        ),
        create_question(
            "Manager says: 'If we double the team size, we'll finish in half the time.' What's the flaw in this logic?",
            [
                "This is always true",
                "Ignores communication overhead, ramp-up time, and task dependencies (Brooks's Law)",
                "Should triple the team instead",
                "Team size doesn't matter"
            ],
            1,
            "Brooks's Law: 'Adding manpower to a late software project makes it later.' Reasons: (1) new members need training (reduces productivity temporarily), (2) communication overhead grows quadratically (n(n-1)/2 pairs), (3) some tasks aren't parallelizable (dependencies), (4) coordination costs increase. Doubling team rarely doubles speed, often reduces it. Critical thinking: recognize false linear assumptions, consider system dynamics and constraints.",
            "Hard",
            95
        ),
        create_question(
            "A blog post claims 'Technology X is always better than Technology Y.' What should you question?",
            [
                "Nothing, accept it as fact",
                "The context, trade-offs, use cases, and evidence supporting the claim",
                "The author's name only",
                "Grammar only"
            ],
            1,
            "Critical evaluation of technology claims: (1) what's the context/use case?, (2) what are the trade-offs?, (3) what evidence supports this? (benchmarks? production experience?), (4) who benefits from this claim? (vendor?), (5) are there counter-examples? No technology is universally superior - all have trade-offs. Question absolutes ('always', 'never', 'best'). Consider: performance, complexity, cost, team expertise, ecosystem maturity.",
            "Medium",
            85
        ),
        create_question(
            "Two metrics show contradictory trends: user signups are up 50%, but revenue is down 20%. What might explain this?",
            [
                "Impossible, ignore one metric",
                "Quality of signups changed (lower-paying users), pricing changes, free tier growth, or conversion rate drop",
                "Metrics are broken",
                "Revenue doesn't matter"
            ],
            1,
            "Contradictory metrics require investigation: (1) segment analysis (who are new signups?), (2) cohort analysis (do new users behave differently?), (3) pricing changes, (4) free vs paid ratio, (5) conversion rate trends. Possible causes: viral growth in low-value segment, competitors targeting high-value users, product changes affecting monetization. Critical thinking: don't cherry-pick metrics, investigate correlations, understand the full picture.",
            "Hard",
            90
        ),
        create_question(
            "A study shows developers using IDE X are 30% more productive. Can you conclude IDE X causes increased productivity?",
            [
                "Yes, definitely",
                "No - correlation doesn't imply causation; could be selection bias or confounding factors",
                "Yes, if the study used statistics",
                "No studies are reliable"
            ],
            1,
            "Correlation ≠ causation. Potential explanations: (1) selection bias (better developers choose IDE X), (2) experience (senior devs use X), (3) confounding factors (X users have better hardware), (4) reverse causation (productive devs can afford/learn X). Need: randomized controlled trials or careful statistical controls. Critical thinking: identify alternative explanations, recognize biases. Mere correlation can't establish causality without ruling out confounders.",
            "Hard",
            95
        ),
        create_question(
            "Someone argues: 'We should use microservices because Google uses them.' What's the logical flaw?",
            [
                "No flaw, always copy Google",
                "Appeal to authority and ignoring context - Google's scale/needs differ from most organizations",
                "Google is wrong",
                "Microservices never work"
            ],
            1,
            "Logical fallacy: appeal to authority + ignoring context. Google's challenges (billions of users, thousands of developers) differ from most companies. Microservices add complexity - worth it at scale, overkill for small teams. Critical thinking: (1) understand your context, (2) evaluate trade-offs for YOUR situation, (3) don't cargo cult practices from different contexts. Good: 'X works at Google because of Y, do we have Y?' Bad: 'X works at Google, so we should do X.'",
            "Hard",
            95
        ),
        create_question(
            "You notice feature usage dropped after a redesign. Manager says 'Users will adapt, give it time.' How should you respond?",
            [
                "Agree and wait indefinitely",
                "Gather user feedback, analyze drop patterns, A/B test if possible, set decision timeline",
                "Roll back immediately without data",
                "Ignore users completely"
            ],
            1,
            "Test assumptions with data: (1) quantify the drop (how much? which segments?), (2) gather qualitative feedback (surveys, support tickets), (3) understand why (confusion? missing features? preference?), (4) A/B test (keep old version for comparison), (5) set decision criteria (if X after Y time, then rollback). Some adaptation is normal, but sustained drops signal problems. Critical thinking: balance patience with responsiveness, use data not opinions.",
            "Medium",
            90
        ),
        create_question(
            "A vendor claims their tool will 'reduce bugs by 90%.' What questions should you ask?",
            [
                "Accept the claim immediately",
                "Evidence source, methodology, context, definition of 'bugs', comparison baseline",
                "No questions needed",
                "Only ask about price"
            ],
            1,
            "Evaluate extraordinary claims: (1) what's the evidence? (case studies? controlled experiments?), (2) what's the methodology? (how measured?), (3) what's the baseline? (90% vs what?), (4) what's the context? (worked for whom? what domain?), (5) how do they define 'bugs'? (severity?). Vendors incentivized to oversell. Look for: peer-reviewed studies, multiple independent sources, realistic claims. Critical thinking: extraordinary claims require extraordinary evidence.",
            "Hard",
            90
        ),
        create_question(
            "Root cause analysis reveals 'human error' as the cause of an outage. Is this a satisfactory conclusion?",
            [
                "Yes, fire the person responsible",
                "No - should investigate systemic issues that enabled the error (poor tooling, unclear processes, missing safeguards)",
                "Yes, humans are always the problem",
                "Ignore the outage"
            ],
            1,
            "'Human error' is rarely the root cause - it's a symptom of systemic issues. Dig deeper with Five Whys: Why did they make the error? (unclear procedure) Why unclear? (not documented) Why not documented? (no process for documentation) Why no process? Blameless post-mortems focus on: (1) what systems failed to prevent error?, (2) how can we make it impossible/harder to repeat?, (3) automation, safeguards, documentation. Critical thinking: systems thinking over individual blame.",
            "Hard",
            100
        ),
        create_question(
            "A report shows your app is #1 on a 'Top 10 Apps' list. What should you verify before sharing this achievement?",
            [
                "Share immediately without checking",
                "Who created the list, criteria used, sample size, potential bias or payment for inclusion",
                "The font used in the report",
                "Nothing, rankings are always objective"
            ],
            1,
            "Verify credibility of accolades: (1) who created it? (reputable source or pay-to-play?), (2) methodology (what criteria? sample size?), (3) selection bias (how were nominees chosen?), (4) was it paid/sponsored?, (5) when published (current or outdated?). Many 'awards' are marketing schemes. Critical thinking: distinguish legitimate recognition from promotional content. Verify before amplifying claims.",
            "Medium",
            85
        ),
        create_question(
            "Data shows users from source A have 2x higher conversion than source B. Should you cut budget from B and invest in A?",
            [
                "Yes, immediately",
                "Not necessarily - need to check: sample size, user quality vs quantity, long-term value, and whether A can scale",
                "No, never change budgets",
                "Flip a coin"
            ],
            1,
            "Avoid hasty conclusions: (1) statistical significance (is sample size adequate?), (2) short vs long-term value (higher churn in A?), (3) scalability (can A handle more volume? diminishing returns?), (4) cost per conversion (A might be more expensive), (5) strategic value (B might target important segment). Critical thinking: consider full picture, long-term effects, constraints. Optimize holistically, not on single metric.",
            "Hard",
            90
        ),
        create_question(
            "You read: 'Developers who use TypeScript make 40% fewer bugs.' What confounding factors might explain this?",
            [
                "None, TypeScript directly causes fewer bugs",
                "Developer experience, project complexity, team practices, code review rigor, testing culture",
                "TypeScript is magic",
                "The study must be wrong"
            ],
            1,
            "Confounding factors: (1) selection (who chooses TypeScript? experienced devs?), (2) project maturity (TS used in newer, better-designed projects?), (3) team practices (teams adopting TS might also do better testing), (4) complexity (TS projects might be different domains), (5) code review culture. To establish causality need: randomized assignment or statistical controls. Critical thinking: identify lurking variables, alternative explanations.",
            "Hard",
            95
        ),
        create_question(
            "Proposal: 'Let's rewrite everything in Technology X because it's newer and better.' What's the problem with this reasoning?",
            [
                "No problem, new is always better",
                "Assumes new = better, ignores rewrite costs/risks, and doesn't identify actual problems being solved",
                "Old technology is always better",
                "Technology doesn't matter"
            ],
            1,
            "Question the premise: (1) what problems does current system have?, (2) will rewrite solve them or introduce new problems?, (3) what's the cost? (time, risk, opportunity cost), (4) can we incrementally improve instead?, (5) is team experienced in X? Rewrites often fail, take longer than estimated, introduce new bugs. New != better. Critical thinking: identify actual problems first, evaluate solutions against problems, consider costs/risks realistically. Strangler fig pattern > big rewrite.",
            "Hard",
            95
        ),
        create_question(
            "Metric dashboard shows all green (targets met). Should you conclude everything is fine?",
            [
                "Yes, celebrate and relax",
                "Not necessarily - check if metrics still align with goals, if gaming is happening, and if leading indicators show future problems",
                "No, always panic",
                "Ignore all metrics"
            ],
            1,
            "Question metrics: (1) do they still measure what matters? (goals changed?), (2) Goodhart's Law: when measure becomes target, it ceases to be good measure (gaming?), (3) are they lagging indicators? (problem brewing not yet visible?), (4) are we missing important signals? Green doesn't mean perfect - might mean wrong metrics. Critical thinking: metrics are proxies not goals, can be gamed, need regular review. Look beyond dashboard to reality.",
            "Medium",
            85
        ),
        create_question(
            "A popular influencer recommends an architecture pattern. Should you adopt it for your project?",
            [
                "Yes, influencers are always right",
                "Evaluate based on your context, requirements, and trade-offs, not popularity",
                "No, never trust influencers",
                "Architecture doesn't matter"
            ],
            1,
            "Evaluate independently: (1) what problem does it solve?, (2) do we have that problem?, (3) what are trade-offs?, (4) what's our context? (team size, scale, domain), (5) what's the evidence? (production use? only tutorials?). Influencers may lack your context, may oversimplify, or promote sponsors. Critical thinking: evaluate claims on merit, not popularity. Understand WHY before adopting WHAT. Context matters more than authority.",
            "Hard",
            90
        )
    ]
    return questions


if __name__ == "__main__":
    db = QuestionDatabase()

    print("Populating Problem Solving questions...")
    db.add_bulk_questions("Problem Solving", populate_problem_solving())
    print(f"✓ Added {len(populate_problem_solving())} Problem Solving questions")

    print("Populating Critical Thinking questions...")
    db.add_bulk_questions("Critical Thinking", populate_critical_thinking())
    print(f"✓ Added {len(populate_critical_thinking())} Critical Thinking questions")

    stats = db.get_statistics()
    print(f"\n{'='*60}")
    print(f"BATCH 5 COMPLETE - Cognitive Skills")
    print(f"{'='*60}")
    print(f"Total questions in database: {stats['total_questions']}")
    print("\nBatch 5 questions by category:")
    for category in ["Problem Solving", "Critical Thinking"]:
        count = db.get_question_count(category)
        print(f"  {category}: {count} questions")
    print(f"\nDatabase saved to: questions_db.json")
    print(f"\n{'='*60}")
    print(f"🎉 ALL BATCHES COMPLETE! 🎉")
    print(f"{'='*60}")
    print(f"\nFinal Statistics:")
    print(f"Total Questions: {stats['total_questions']}")
    print(f"\nAll {len(db.get_all_categories())} categories populated!")
    for cat in db.get_all_categories():
        count = db.get_question_count(cat)
        if count > 0:
            print(f"  ✓ {cat}: {count} questions")
