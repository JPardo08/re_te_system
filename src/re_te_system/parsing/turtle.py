"""Conservative RDF Turtle parsing without output repair."""

from __future__ import annotations

from typing import Mapping

from re_te_system.contracts import ParseIssue, ParseResult, ParsedTriple


def parse_turtle(
    text: str,
    segment_id: str | None = None,
    *,
    prefixes: Mapping[str, str] | None = None,
) -> ParseResult:
    """Parse exact Turtle text while preserving parser emission order and duplicates."""
    if not text.strip():
        return ParseResult(
            (),
            (ParseIssue("EMPTY_GRAPH", "Turtle output is empty", "soft"),),
        )

    try:
        from rdflib import BNode, Graph, Literal, URIRef
    except ImportError as exc:
        raise ImportError('Install Turtle support with "pip install -e .[pythia]"') from exc

    class RecordingGraph(Graph):
        def __init__(self) -> None:
            super().__init__()
            self.emitted: list[tuple[object, object, object]] = []

        def add(self, triple: tuple[object, object, object]):  # type: ignore[override]
            self.emitted.append(triple)
            return super().add(triple)

    graph = RecordingGraph()
    prefix_context = []
    for prefix, namespace in sorted((prefixes or {}).items()):
        graph.bind(prefix, URIRef(namespace), replace=True)
        prefix_context.append(f"@prefix {prefix}: <{namespace}> .")
    parser_input = "\n".join((*prefix_context, text))

    try:
        graph.parse(data=parser_input, format="turtle")
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        code = (
            "PREFIX_RESOLUTION_FAILURE"
            if "prefix" in lowered
            and ("not bound" in lowered or "unknown" in lowered or "not found" in lowered)
            else "INVALID_TURTLE"
        )
        return ParseResult((), (ParseIssue(code, message, "hard"),))

    if not graph.emitted:
        return ParseResult(
            (),
            (ParseIssue("EMPTY_GRAPH", "Turtle parsed successfully but emitted no triples"),),
        )

    blank_nodes: dict[object, str] = {}

    def term_text(term: object) -> str:
        if isinstance(term, BNode):
            if term not in blank_nodes:
                blank_nodes[term] = f"_:b{len(blank_nodes)}"
            return blank_nodes[term]
        return str(term)

    triples: list[ParsedTriple] = []
    issues: list[ParseIssue] = []
    for index, (subject, predicate, object_) in enumerate(graph.emitted):
        predicate_is_uri = isinstance(predicate, URIRef)
        if not predicate_is_uri:
            issues.append(
                ParseIssue(
                    "NON_URI_PREDICATE",
                    "RDF predicate is not a URI",
                    "hard",
                    index,
                )
            )
        literal = object_ if isinstance(object_, Literal) else None
        triples.append(
            ParsedTriple(
                subject=term_text(subject),
                relation=term_text(predicate),
                object=term_text(object_),
                segment_id=segment_id,
                subject_is_uri=isinstance(subject, URIRef),
                predicate_is_uri=predicate_is_uri,
                object_is_uri=isinstance(object_, URIRef),
                object_is_literal=literal is not None,
                datatype=str(literal.datatype) if literal and literal.datatype else None,
                language=literal.language if literal else None,
            )
        )
    return ParseResult(tuple(triples), tuple(issues))
