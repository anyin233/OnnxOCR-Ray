import typer
import uvicorn
import socket
from enum import Enum
import os
import signal

app = typer.Typer(help="OCR Microservice Unify Entry Point")


class ServiceType(str, Enum):
  classification = "classification"
  detection = "detection"
  recognition = "recognition"
  # Add more services as needed

def find_neareast_available_port(port: int) -> int:
  """Find the nearest available port starting from the given port."""
  while True:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
          if s.connect_ex(('localhost', port)) != 0:
              return port
      port += 1


def add_common_method(app):
  @app.get("/stop", summary="Stop the service")
  def stop_service():
    """Stop the service gracefully."""
    typer.echo("Stopping the service...")
    os.kill(os.getpid(), signal.SIGINT)
    return {"message": "Service is stopping"}

def run_service(service: ServiceType, port: int, host: str):
  """Run a specific service in the background."""
  if service == ServiceType.classification:
    from microservices import classification_app
    add_common_method(classification_app)
    uvicorn.run(classification_app, host=host, port=port)
  elif service == ServiceType.detection:
    from microservices import detection_app
    add_common_method(detection_app)
    uvicorn.run(detection_app, host=host, port=port)
  elif service == ServiceType.recognition:
    from microservices import recognition_app
    add_common_method(recognition_app)
    uvicorn.run(recognition_app, host=host, port=port)
  else:
    typer.echo(f"Service {service} is not implemented yet.")
    raise typer.Exit(code=1)

@app.command(name="run", help="Run the OCR microservice")
def main(
  service: ServiceType = typer.Option(..., "--service", "-s", help="Service to run"),
  port: int = typer.Option(5005, "--port", "-p", help="Port to run the service on"),
  host: str = typer.Option("0.0.0.0", "--host", "-H", help="Host to run the service on")
):
  """Main entry point for OCR microservices"""
  
  port = find_neareast_available_port(port)
  typer.echo(f"Starting {service} service on {host}:{port}")
  run_service(service, port, host)
  pass

if __name__ == "__main__":
  app()
